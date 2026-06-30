require("@nomicfoundation/hardhat-toolbox");
require("dotenv").config();

/** @type import('hardhat/config').HardhatUserConfig */
module.exports = {
  solidity: {
    version: "0.8.19",
    settings: {
      optimizer: { enabled: true, runs: 200 },
    },
  },

  networks: {
    // Local testing
    hardhat: {},

    // Polygon Mumbai testnet
    mumbai: {
      url:      process.env.POLYGON_RPC_URL || "https://rpc-mumbai.maticvigil.com",
      accounts: process.env.WALLET_PRIVATE_KEY ? [process.env.WALLET_PRIVATE_KEY] : [],
      chainId:  80001,
    },

    // Polygon mainnet (use after testing)
    polygon: {
      url:      "https://polygon-rpc.com",
      accounts: process.env.WALLET_PRIVATE_KEY ? [process.env.WALLET_PRIVATE_KEY] : [],
      chainId:  137,
    },
  },

  etherscan: {
    apiKey: {
      polygonMumbai: process.env.POLYGONSCAN_API_KEY || "",
      polygon:       process.env.POLYGONSCAN_API_KEY || "",
    },
  },
};
