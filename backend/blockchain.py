"""
Blockchain Client
Handles all Web3 interactions with the DeepfakeAudit smart contract
on Polygon (Mumbai testnet).

Functions:
  submit_detection()  → write detection result to chain
  verify_file()       → query chain for file's detection history
  get_record()        → get a specific record by ID
  get_file_stats()    → aggregate stats for a file hash
"""

import json
import os
from typing import Dict, Optional
from pathlib import Path

try:
    from web3 import Web3
    from web3.middleware import geth_poa_middleware
    WEB3_AVAILABLE = True
except ImportError:
    WEB3_AVAILABLE = False


# ABI for DeepfakeAudit.sol — only the functions we call
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
    {
        "inputs":  [{"name": "fileHash", "type": "string"}],
        "name":    "getRecordsByHash",
        "outputs": [{"name": "", "type": "uint256[]"}],
        "stateMutability": "view",
        "type": "function",
    },
    {
        "inputs":  [{"name": "fileHash", "type": "string"}],
        "name":    "verifyFile",
        "outputs": [
            {"name": "analyzed",         "type": "bool"},
            {"name": "latestVerdict",    "type": "bool"},
            {"name": "latestTimestamp",  "type": "uint256"},
            {"name": "analysisCount",    "type": "uint256"},
        ],
        "stateMutability": "view",
        "type": "function",
    },
    {
        "inputs":  [{"name": "fileHash", "type": "string"}],
        "name":    "getFileStats",
        "outputs": [
            {"name": "totalAnalyses",        "type": "uint256"},
            {"name": "fakeVerdicts",         "type": "uint256"},
            {"name": "realVerdicts",         "type": "uint256"},
            {"name": "avgFakeProbBasisPts",  "type": "uint256"},
        ],
        "stateMutability": "view",
        "type": "function",
    },
    {
        "anonymous": False,
        "inputs": [
            {"indexed": True,  "name": "recordId",        "type": "uint256"},
            {"indexed": True,  "name": "fileHash",         "type": "string"},
            {"indexed": True,  "name": "submitter",        "type": "address"},
            {"indexed": False, "name": "isFake",           "type": "bool"},
            {"indexed": False, "name": "fakeProbability",  "type": "uint16"},
            {"indexed": False, "name": "timestamp",        "type": "uint256"},
        ],
        "name": "DetectionSubmitted",
        "type": "event",
    },
]


class BlockchainClient:
    """
    Manages connection to Polygon and interaction with DeepfakeAudit contract.
    Gracefully degrades if web3 is unavailable or no private key is configured.
    """

    def __init__(
        self,
        rpc_url:          str,
        contract_address: str,
        private_key:      str,
    ):
        self.rpc_url          = rpc_url
        self.contract_address = contract_address
        self.private_key      = private_key
        self._w3              = None
        self._contract        = None
        self._account         = None
        self._enabled         = WEB3_AVAILABLE and bool(contract_address) and bool(private_key)

        if self._enabled:
            self._init_web3()

    def _init_web3(self):
        try:
            self._w3 = Web3(Web3.HTTPProvider(self.rpc_url))
            # Polygon uses PoA — inject middleware
            self._w3.middleware_onion.inject(geth_poa_middleware, layer=0)

            self._account  = self._w3.eth.account.from_key(self.private_key)
            self._contract = self._w3.eth.contract(
                address = Web3.to_checksum_address(self.contract_address),
                abi     = CONTRACT_ABI,
            )
        except Exception as e:
            self._enabled = False
            print(f"[Blockchain] Init failed (will run in mock mode): {e}")

    def submit_detection(
        self,
        file_hash:        str,
        is_fake:          bool,
        fake_probability: float,
        confidence:       str,
        media_type:       str,
        models_used:      str,
    ) -> str:
        """
        Submits detection result to the smart contract.
        Returns transaction hash.
        """
        # Convert probability to basis points (0–10000)
        fake_prob_bp = int(round(fake_probability * 10000))
        fake_prob_bp = max(0, min(10000, fake_prob_bp))

        if not self._enabled:
            return self._mock_tx_hash(file_hash)

        try:
            nonce = self._w3.eth.get_transaction_count(self._account.address)

            tx = self._contract.functions.submitDetection(
                file_hash,
                is_fake,
                fake_prob_bp,
                confidence,
                media_type,
                models_used,
            ).build_transaction({
                "from":     self._account.address,
                "nonce":    nonce,
                "gas":      200_000,
                "gasPrice": self._w3.eth.gas_price,
            })

            signed = self._account.sign_transaction(tx)
            tx_hash = self._w3.eth.send_raw_transaction(signed.rawTransaction)
            receipt = self._w3.eth.wait_for_transaction_receipt(tx_hash, timeout=120)

            return receipt.transactionHash.hex()

        except Exception as e:
            raise RuntimeError(f"Blockchain submission failed: {e}")

    def verify_file(self, file_hash: str) -> Dict:
        """
        Queries the contract for a file's detection history.
        Returns structured dict for the frontend.
        """
        if not self._enabled:
            return self._mock_verify(file_hash)

        try:
            analyzed, latest_verdict, latest_ts, analysis_count = (
                self._contract.functions.verifyFile(file_hash).call()
            )

            if not analyzed:
                return {
                    "analyzed":       False,
                    "file_hash":      file_hash,
                    "message":        "This file has not been analyzed yet.",
                }

            # Also get aggregate stats
            total, fakes, reals, avg_prob_bp = (
                self._contract.functions.getFileStats(file_hash).call()
            )

            # Get all record IDs for detailed history
            record_ids = self._contract.functions.getRecordsByHash(file_hash).call()
            history    = []

            for rid in record_ids[-5:]:  # Last 5 analyses
                rec = self._contract.functions.getRecord(rid).call()
                history.append({
                    "record_id":        rid,
                    "verdict":          "FAKE" if rec[1] else "REAL",
                    "fake_probability": round(rec[2] / 100, 2),
                    "confidence":       rec[3],
                    "media_type":       rec[4],
                    "models_used":      rec[5],
                    "submitter":        rec[6],
                    "timestamp":        rec[7],
                    "block_number":     rec[8],
                })

            return {
                "analyzed":              True,
                "file_hash":             file_hash,
                "latest_verdict":        "FAKE" if latest_verdict else "REAL",
                "latest_timestamp":      latest_ts,
                "total_analyses":        total,
                "fake_verdicts":         fakes,
                "real_verdicts":         reals,
                "avg_fake_probability":  round(avg_prob_bp / 100, 2),
                "detection_history":     history,
                "explorer_url":          f"https://mumbai.polygonscan.com/address/{self.contract_address}",
            }

        except Exception as e:
            raise RuntimeError(f"Blockchain query failed: {e}")

    def _mock_tx_hash(self, file_hash: str) -> str:
        """Returns a deterministic mock tx hash when web3 is unavailable."""
        import hashlib
        return "0x" + hashlib.sha256(file_hash.encode()).hexdigest()

    def _mock_verify(self, file_hash: str) -> Dict:
        """Mock response for development without blockchain."""
        return {
            "analyzed":  False,
            "file_hash": file_hash,
            "message":   "Blockchain not configured (mock mode). Set POLYGON_RPC_URL, CONTRACT_ADDRESS, and WALLET_PRIVATE_KEY.",
            "mock":      True,
        }
