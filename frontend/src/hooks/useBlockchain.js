/**
 * useBlockchain hook
 * Manages MetaMask wallet connection and smart contract interaction
 * on Polygon Amoy testnet (Mumbai was deprecated April 2024).
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

// Polygon Amoy testnet (official replacement for deprecated Mumbai)
const POLYGON_AMOY = {
  chainId:           "0x13882", // 80002 in hex
  chainName:         "Polygon Amoy Testnet",
  nativeCurrency:    { name: "POL", symbol: "POL", decimals: 18 },
  rpcUrls:           ["https://rpc-amoy.polygon.technology"],
  blockExplorerUrls: ["https://amoy.polygonscan.com"],
};

export function useBlockchain() {
  const [account,   setAccount]   = useState(null);
  const [connected, setConnected] = useState(false);
  const [loading,   setLoading]   = useState(false);
  const [error,     setError]     = useState(null);

  const connect = useCallback(async () => {
    setError(null);
    setLoading(true);
    try {
      if (!window.ethereum) throw new Error("MetaMask not found. Please install it.");

      const accounts = await window.ethereum.request({ method: "eth_requestAccounts" });

      try {
        await window.ethereum.request({
          method: "wallet_switchEthereumChain",
          params: [{ chainId: POLYGON_AMOY.chainId }],
        });
      } catch (switchErr) {
        if (switchErr.code === 4902) {
          await window.ethereum.request({
            method: "wallet_addEthereumChain",
            params: [POLYGON_AMOY],
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

  const getContract = useCallback(async (readOnly = false) => {
    if (!window.ethereum) throw new Error("MetaMask not available");
    const provider = new ethers.BrowserProvider(window.ethereum);
    if (readOnly) {
      return new ethers.Contract(CONTRACT_ADDRESS, CONTRACT_ABI, provider);
    }
    const signer = await provider.getSigner();
    return new ethers.Contract(CONTRACT_ADDRESS, CONTRACT_ABI, signer);
  }, []);

  const submitDetection = useCallback(async ({
    fileHash, isFake, fakeProbability, confidence, mediaType, modelsUsed,
  }) => {
    if (!connected) throw new Error("Wallet not connected");
    if (!CONTRACT_ADDRESS) throw new Error("Contract address not configured");

    setLoading(true);
    setError(null);
    try {
      const contract = await getContract(false);
      const probBP   = Math.round(fakeProbability * 10000);
      const tx       = await contract.submitDetection(
        fileHash, isFake, probBP, confidence, mediaType, modelsUsed
      );
      const receipt  = await tx.wait();
      return {
        txHash:      receipt.hash,
        explorerUrl: `https://amoy.polygonscan.com/tx/${receipt.hash}`,
        blockNumber: receipt.blockNumber,
      };
    } catch (e) {
      setError(e.message);
      throw e;
    } finally {
      setLoading(false);
    }
  }, [connected, getContract]);

  const verifyOnChain = useCallback(async (fileHash) => {
    if (!CONTRACT_ADDRESS) return null;
    try {
      const contract = await getContract(true);
      const [analyzed, latestVerdict, latestTimestamp, analysisCount] =
        await contract.verifyFile(fileHash);
      return {
        analyzed,
        verdict:       latestVerdict ? "FAKE" : "REAL",
        timestamp:     Number(latestTimestamp),
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