"""
Blockchain Client
Handles Web3 interactions with the DeepfakeAudit smart contract
on Polygon Amoy testnet (chainId 80002).

Note: Mumbai testnet was deprecated April 2024 — Amoy is the official replacement.
"""

import os
from typing import Dict

try:
    from web3 import Web3
    WEB3_AVAILABLE = True
except ImportError:
    WEB3_AVAILABLE = False


CONTRACT_ABI = [
    {
        "inputs": [
            {"name": "fileHash",         "type": "string"},
            {"name": "isFake",           "type": "bool"},
            {"name": "fakeProbBasisPts", "type": "uint16"},
            {"name": "confidence",       "type": "string"},
            {"name": "mediaType",        "type": "string"},
            {"name": "modelsUsed",       "type": "string"},
        ],
        "name":    "submitDetection",
        "outputs": [{"name": "recordId", "type": "uint256"}],
        "stateMutability": "nonpayable",
        "type":    "function",
    },
    {
        "inputs":  [{"name": "fileHash", "type": "string"}],
        "name":    "verifyFile",
        "outputs": [
            {"name": "analyzed",        "type": "bool"},
            {"name": "latestVerdict",   "type": "bool"},
            {"name": "latestTimestamp", "type": "uint256"},
            {"name": "analysisCount",   "type": "uint256"},
        ],
        "stateMutability": "view",
        "type": "function",
    },
    {
        "inputs":  [{"name": "fileHash", "type": "string"}],
        "name":    "getFileStats",
        "outputs": [
            {"name": "totalAnalyses",       "type": "uint256"},
            {"name": "fakeVerdicts",        "type": "uint256"},
            {"name": "realVerdicts",        "type": "uint256"},
            {"name": "avgFakeProbBasisPts", "type": "uint256"},
        ],
        "stateMutability": "view",
        "type": "function",
    },
    {
        "inputs":  [{"name": "fileHash", "type": "string"}],
        "name":    "getRecordsByHash",
        "outputs": [{"name": "", "type": "uint256[]"}],
        "stateMutability": "view",
        "type": "function",
    },
    {
        "inputs":  [{"name": "recordId", "type": "uint256"}],
        "name":    "getRecord",
        "outputs": [
            {
                "components": [
                    {"name": "fileHash",        "type": "string"},
                    {"name": "isFake",          "type": "bool"},
                    {"name": "fakeProbability", "type": "uint16"},
                    {"name": "confidence",      "type": "string"},
                    {"name": "mediaType",       "type": "string"},
                    {"name": "modelsUsed",      "type": "string"},
                    {"name": "submitter",       "type": "address"},
                    {"name": "timestamp",       "type": "uint256"},
                    {"name": "blockNumber",     "type": "uint256"},
                ],
                "type": "tuple",
            }
        ],
        "stateMutability": "view",
        "type": "function",
    },
]

AMOY_EXPLORER = "https://amoy.polygonscan.com"


class BlockchainClient:
    """
    Manages connection to Polygon Amoy and interaction with DeepfakeAudit.
    Gracefully degrades to mock mode if web3 unavailable or unconfigured.
    """

    def __init__(self, rpc_url: str, contract_address: str, private_key: str):
        self.rpc_url          = rpc_url or "https://rpc-amoy.polygon.technology"
        self.contract_address = contract_address
        self.private_key      = private_key
        self._w3       = None
        self._contract = None
        self._account  = None
        self._enabled  = WEB3_AVAILABLE and bool(contract_address) and bool(private_key)

        if self._enabled:
            self._init_web3()

    def _init_web3(self):
        try:
            self._w3 = Web3(Web3.HTTPProvider(self.rpc_url))
            self._account  = self._w3.eth.account.from_key(self.private_key)
            self._contract = self._w3.eth.contract(
                address = Web3.to_checksum_address(self.contract_address),
                abi     = CONTRACT_ABI,
            )
            print(f"[Blockchain] Connected to Amoy. Account: {self._account.address}")
        except Exception as e:
            self._enabled = False
            print(f"[Blockchain] Init failed (mock mode): {e}")

    def submit_detection(self, file_hash, is_fake, fake_probability,
                          confidence, media_type, models_used) -> str:
        fake_prob_bp = max(0, min(10000, int(round(fake_probability * 10000))))

        if not self._enabled:
            return self._mock_tx_hash(file_hash)

        try:
            nonce = self._w3.eth.get_transaction_count(self._account.address)
            tx = self._contract.functions.submitDetection(
                file_hash, is_fake, fake_prob_bp, confidence, media_type, models_used
            ).build_transaction({
                "from":     self._account.address,
                "nonce":    nonce,
                "gas":      200_000,
                "gasPrice": self._w3.eth.gas_price,
                "chainId":  80002,  # Amoy
            })
            signed  = self._account.sign_transaction(tx)
            tx_hash = self._w3.eth.send_raw_transaction(signed.raw_transaction)
            receipt = self._w3.eth.wait_for_transaction_receipt(tx_hash, timeout=120)
            return receipt.transactionHash.hex()
        except Exception as e:
            raise RuntimeError(f"Blockchain submission failed: {e}")

    def verify_file(self, file_hash: str) -> Dict:
        if not self._enabled:
            return self._mock_verify(file_hash)

        try:
            analyzed, latest_verdict, latest_ts, count = (
                self._contract.functions.verifyFile(file_hash).call()
            )
            if not analyzed:
                return {"analyzed": False, "file_hash": file_hash,
                        "message": "This file has not been analyzed yet."}

            total, fakes, reals, avg_bp = (
                self._contract.functions.getFileStats(file_hash).call()
            )
            record_ids = self._contract.functions.getRecordsByHash(file_hash).call()
            history = []
            for rid in record_ids[-5:]:
                rec = self._contract.functions.getRecord(rid).call()
                history.append({
                    "record_id": rid, "verdict": "FAKE" if rec[1] else "REAL",
                    "fake_probability": round(rec[2]/100, 2), "confidence": rec[3],
                    "media_type": rec[4], "models_used": rec[5],
                    "submitter": rec[6], "timestamp": rec[7], "block_number": rec[8],
                })

            return {
                "analyzed": True, "file_hash": file_hash,
                "latest_verdict": "FAKE" if latest_verdict else "REAL",
                "latest_timestamp": latest_ts, "total_analyses": total,
                "fake_verdicts": fakes, "real_verdicts": reals,
                "avg_fake_probability": round(avg_bp/100, 2),
                "detection_history": history,
                "explorer_url": f"{AMOY_EXPLORER}/address/{self.contract_address}",
            }
        except Exception as e:
            raise RuntimeError(f"Blockchain query failed: {e}")

    def _mock_tx_hash(self, file_hash: str) -> str:
        import hashlib
        return "0x" + hashlib.sha256(file_hash.encode()).hexdigest()

    def _mock_verify(self, file_hash: str) -> Dict:
        return {
            "analyzed": False, "file_hash": file_hash, "mock": True,
            "message": "Blockchain not configured. Set POLYGON_RPC_URL, CONTRACT_ADDRESS, WALLET_PRIVATE_KEY in .env",
        }