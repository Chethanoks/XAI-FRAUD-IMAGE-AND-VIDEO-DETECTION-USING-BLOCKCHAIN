const { expect } = require("chai");
const { ethers }  = require("hardhat");

describe("DeepfakeAudit", function () {
  let contract, owner, user1;

  const FAKE_HASH  = "a".repeat(64);   // valid 64-char hex string
  const REAL_HASH  = "b".repeat(64);

  beforeEach(async () => {
    [owner, user1] = await ethers.getSigners();
    const Factory  = await ethers.getContractFactory("DeepfakeAudit");
    contract       = await Factory.deploy();
    await contract.waitForDeployment();
  });

  it("should submit and retrieve a FAKE detection", async () => {
    await contract.submitDetection(FAKE_HASH, true, 8750, "High", "image", "CLIP,SBI");

    const [analyzed, verdict, , count] = await contract.verifyFile(FAKE_HASH);
    expect(analyzed).to.equal(true);
    expect(verdict).to.equal(true);   // isFake = true
    expect(count).to.equal(1n);
  });

  it("should submit and retrieve a REAL detection", async () => {
    await contract.submitDetection(REAL_HASH, false, 1200, "Very High", "video", "AltFreezing,LipForensics");

    const [analyzed, verdict] = await contract.verifyFile(REAL_HASH);
    expect(analyzed).to.equal(true);
    expect(verdict).to.equal(false);  // isFake = false
  });

  it("should return not analyzed for unknown hash", async () => {
    const [analyzed] = await contract.verifyFile("c".repeat(64));
    expect(analyzed).to.equal(false);
  });

  it("should accumulate multiple records for same file", async () => {
    await contract.submitDetection(FAKE_HASH, true,  8000, "High",   "image", "CLIP");
    await contract.submitDetection(FAKE_HASH, false, 4000, "Medium", "image", "CLIP");

    const [total, fakes, reals] = await contract.getFileStats(FAKE_HASH);
    expect(total).to.equal(2n);
    expect(fakes).to.equal(1n);
    expect(reals).to.equal(1n);
  });

  it("should emit DetectionSubmitted event", async () => {
    await expect(
      contract.submitDetection(FAKE_HASH, true, 9000, "Very High", "image", "CLIP,SBI")
    ).to.emit(contract, "DetectionSubmitted");
  });

  it("should reject invalid hash length", async () => {
    await expect(
      contract.submitDetection("tooshort", true, 5000, "High", "image", "CLIP")
    ).to.be.revertedWith("Invalid SHA-256 hash length");
  });

  it("should reject probability > 10000", async () => {
    await expect(
      contract.submitDetection(FAKE_HASH, true, 10001, "High", "image", "CLIP")
    ).to.be.revertedWith("Probability must be 0–10000");
  });

  it("should track records per submitter", async () => {
    await contract.connect(user1).submitDetection(FAKE_HASH, true, 7500, "High", "image", "CLIP");
    const ids = await contract.getRecordsBySubmitter(user1.address);
    expect(ids.length).to.equal(1);
  });

  it("should increment totalRecords correctly", async () => {
    expect(await contract.totalRecords()).to.equal(0n);
    await contract.submitDetection(FAKE_HASH, true,  8000, "High", "image", "CLIP");
    await contract.submitDetection(REAL_HASH, false, 2000, "High", "video", "AltFreezing");
    expect(await contract.totalRecords()).to.equal(2n);
  });
});
