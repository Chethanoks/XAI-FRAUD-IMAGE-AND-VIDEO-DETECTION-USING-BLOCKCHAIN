/**
 * useBlockchain hook
 * Manages MetaMask wallet connection and smart contract interaction.
 */
import { useState, useCallback } from "react";
import { ethers } from "ethers";

const CONTRACT_ADDRESS = import.meta.env.VITE_CONTRACT_ADDRESS || "";

const CONTRACT_ABI = [
  "function submitDetection(string fileHash, bool isFake, uint16 fakeProbBasisPts, string confidence, string mediaType, string modelsUsed) returns (uint256)",
  "function verifyFile(string fileHash) view returns (bool analyzed, bool latestVerdict, uint256 latestTimestamp, uint256 analysisCount)",
  "function getRecordsByHash(string fileHash) view returns (uint256[])",
  "function getRecord(uint256 recordId) view returns (tuple(string fileHash, bool isFake, uint16 fakeProbability, string confidence, string mediaType, string modelsUsed, address submitter, uint256 timestamp, uint256 blockNumber))",
  "event DetectionSubmitted(uint256 indexed recordId, string indexed fileHash, address indexed submitter, bool isFake, uint16 fakeProbability, uint256 timestamp)",
];

// Polygon Mumbai testnet
const POLYGON_MUMBAI = {
  chainId:         "0x13881",
  chainName:       "Polygon Mumbai",
  nativeCurrency:  { name: "MATIC", symbol: "MATIC", decimals: 18 },
  rpcUrls:         ["https://rpc-mumbai.maticvigil.com"],
  blockExplorerUrls: ["https://mumbai.polygonscan.com"],
};

export function useBlockchain() {
  const [account,   setAccount]   = useState(null);
  const [connected, setConnected] = useState(false);
  const [loading,   setLoading]   = useState(false);
  const [error,     setError]     = useState(null);

  // ─── Connect wallet ────────────────────────────────────────────────────────

  const connect = useCallback(async () => {
    setError(null);
    setLoading(true);
    try {
      if (!window.ethereum) throw new Error("MetaMask not found. Please install it.");

      // Request accounts
      const accounts = await window.ethereum.request({ method: "eth_requestAccounts" });

      // Switch to Polygon Mumbai
      try {
        await window.ethereum.request({
          method: "wallet_switchEthereumChain",
          params: [{ chainId: POLYGON_MUMBAI.chainId }],
        });
      } catch (switchErr) {
        // Chain not added — add it
        if (switchErr.code === 4902) {
          await window.ethereum.request({
            method: "wallet_addEthereumChain",
            params: [POLYGON_MUMBAI],
          });
        }
      }

      setAccount(accounts[0]);
      setConnected(true);
    } catch (e) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  }, []);

  const disconnect = useCallback(() => {
    setAccount(null);
    setConnected(false);
  }, []);

  // ─── Get contract instance ─────────────────────────────────────────────────

  const getContract = useCallback(async (readOnly = false) => {
    if (!window.ethereum) throw new Error("MetaMask not available");
    const provider = new ethers.BrowserProvider(window.ethereum);
    if (readOnly) {
      return new ethers.Contract(CONTRACT_ADDRESS, CONTRACT_ABI, provider);
    }
    const signer = await provider.getSigner();
    return new ethers.Contract(CONTRACT_ADDRESS, CONTRACT_ABI, signer);
  }, []);

  // ─── Submit detection to blockchain ────────────────────────────────────────

  const submitDetection = useCallback(async ({
    fileHash, isFake, fakeProbability, confidence, mediaType, modelsUsed,
  }) => {
    if (!connected) throw new Error("Wallet not connected");
    if (!CONTRACT_ADDRESS) throw new Error("Contract address not configured");

    setLoading(true);
    setError(null);
    try {
      const contract   = await getContract(false);
      const probBP     = Math.round(fakeProbability * 10000);
      const tx         = await contract.submitDetection(
        fileHash, isFake, probBP, confidence, mediaType, modelsUsed
      );
      const receipt    = await tx.wait();
      return {
        txHash:      receipt.hash,
        explorerUrl: `https://mumbai.polygonscan.com/tx/${receipt.hash}`,
        blockNumber: receipt.blockNumber,
      };
    } catch (e) {
      setError(e.message);
      throw e;
    } finally {
      setLoading(false);
    }
  }, [connected, getContract]);

  // ─── Verify file on-chain ──────────────────────────────────────────────────

  const verifyOnChain = useCallback(async (fileHash) => {
    if (!CONTRACT_ADDRESS) return null;
    try {
      const contract = await getContract(true);
      const [analyzed, latestVerdict, latestTimestamp, analysisCount] =
        await contract.verifyFile(fileHash);
      return {
        analyzed,
        verdict:      latestVerdict ? "FAKE" : "REAL",
        timestamp:    Number(latestTimestamp),
        analysisCount: Number(analysisCount),
      };
    } catch {
      return null;
    }
  }, [getContract]);

  return {
    account, connected, loading, error,
    connect, disconnect, submitDetection, verifyOnChain,
  };
}
