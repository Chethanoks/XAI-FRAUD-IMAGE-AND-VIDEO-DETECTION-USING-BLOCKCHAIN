/**
 * Deploy DeepfakeAudit.sol to Polygon Amoy testnet
 * (Mumbai is deprecated as of April 2024 — Amoy is the official replacement)
 *
 * Usage:
 *   cd contracts
 *   npm install
 *   npx hardhat run scripts/deploy.js --network amoy
 *
 * After deployment, copy the contract address to:
 *   - .env → CONTRACT_ADDRESS
 *   - frontend/.env → VITE_CONTRACT_ADDRESS
 */

const { ethers } = require("hardhat");
const fs   = require("fs");
const path = require("path");

async function main() {
  const [deployer] = await ethers.getSigners();

  console.log("Deploying DeepfakeAudit contract to Polygon Amoy...");
  console.log("Deployer address:", deployer.address);

  const balance = await ethers.provider.getBalance(deployer.address);
  console.log("Deployer balance:", ethers.formatEther(balance), "POL");

  if (balance === 0n) {
    throw new Error(
      "Deployer has no test POL. Get free Amoy POL from https://faucet.polygon.technology (select Amoy network)"
    );
  }

  const DeepfakeAudit = await ethers.getContractFactory("DeepfakeAudit");
  const contract      = await DeepfakeAudit.deploy();
  await contract.waitForDeployment();

  const address = await contract.getAddress();
  console.log("\n✓ DeepfakeAudit deployed to:", address);
  console.log("  Network:  Polygon Amoy Testnet (chainId 80002)");
  console.log("  Explorer: https://amoy.polygonscan.com/address/" + address);

  const deployInfo = {
    contractAddress: address,
    deployer:        deployer.address,
    network:         "amoy",
    chainId:         80002,
    deployedAt:      new Date().toISOString(),
  };

  fs.writeFileSync(
    path.join(__dirname, "deployed.json"),
    JSON.stringify(deployInfo, null, 2)
  );

  console.log("\n✓ Deployment info saved to contracts/deployed.json");
  console.log("\nNext steps:");
  console.log("  1. Add to .env:           CONTRACT_ADDRESS=" + address);
  console.log("  2. Add to frontend/.env:  VITE_CONTRACT_ADDRESS=" + address);
  console.log("  3. Verify on Polygonscan: npx hardhat verify --network amoy " + address);
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});