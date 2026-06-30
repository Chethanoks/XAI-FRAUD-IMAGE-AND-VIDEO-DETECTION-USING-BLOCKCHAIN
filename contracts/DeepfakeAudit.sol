// SPDX-License-Identifier: MIT
pragma solidity ^0.8.19;

/**
 * @title DeepfakeAudit
 * @notice Immutable on-chain audit trail for deepfake detection results.
 *
 * Stores:
 *   - SHA-256 hash of the analyzed media file
 *   - Detection verdict (REAL / FAKE)
 *   - Confidence level
 *   - Fake probability (scaled 0–10000 for integer storage)
 *   - Timestamp
 *   - Submitter wallet address
 *   - Media type (image / video)
 *   - Detection models used
 *
 * Anyone can verify a file's detection history by querying its hash.
 * Results are immutable once submitted — no editing or deletion.
 */
contract DeepfakeAudit {

    // -----------------------------------------------------------------------
    // Structs
    // -----------------------------------------------------------------------

    struct DetectionRecord {
        string  fileHash;           // SHA-256 hex string of the media file
        bool    isFake;             // true = FAKE, false = REAL
        uint16  fakeProbability;    // 0–10000 (divide by 100 for percentage)
        string  confidence;         // "Very High" | "High" | "Medium" | "Low"
        string  mediaType;          // "image" | "video"
        string  modelsUsed;         // comma-separated model names
        address submitter;          // wallet address of the submitter
        uint256 timestamp;          // block.timestamp
        uint256 blockNumber;        // block.number for additional verifiability
    }

    // -----------------------------------------------------------------------
    // Storage
    // -----------------------------------------------------------------------

    // recordId → DetectionRecord
    mapping(uint256 => DetectionRecord) public records;

    // fileHash → list of recordIds (one file can be checked multiple times)
    mapping(string => uint256[]) public hashToRecordIds;

    // submitter address → list of recordIds
    mapping(address => uint256[]) public submitterToRecordIds;

    // Auto-incrementing record counter
    uint256 public totalRecords;

    // -----------------------------------------------------------------------
    // Events
    // -----------------------------------------------------------------------

    event DetectionSubmitted(
        uint256 indexed recordId,
        string  indexed fileHash,
        address indexed submitter,
        bool    isFake,
        uint16  fakeProbability,
        uint256 timestamp
    );

    // -----------------------------------------------------------------------
    // Submit
    // -----------------------------------------------------------------------

    /**
     * @notice Submit a detection result to the blockchain.
     * @param fileHash       SHA-256 hash of the analyzed media file (hex string)
     * @param isFake         true if detected as fake
     * @param fakeProbBasisPts fake probability * 10000 (e.g., 85.43% → 8543)
     * @param confidence     confidence label string
     * @param mediaType      "image" or "video"
     * @param modelsUsed     comma-separated list of models
     */
    function submitDetection(
        string  calldata fileHash,
        bool             isFake,
        uint16           fakeProbBasisPts,
        string  calldata confidence,
        string  calldata mediaType,
        string  calldata modelsUsed
    ) external returns (uint256 recordId) {
        require(bytes(fileHash).length == 64,  "Invalid SHA-256 hash length");
        require(fakeProbBasisPts <= 10000,     "Probability must be 0–10000");
        require(bytes(confidence).length > 0,  "Confidence cannot be empty");
        require(bytes(mediaType).length > 0,   "Media type cannot be empty");

        recordId = totalRecords;
        totalRecords++;

        records[recordId] = DetectionRecord({
            fileHash:        fileHash,
            isFake:          isFake,
            fakeProbability: fakeProbBasisPts,
            confidence:      confidence,
            mediaType:       mediaType,
            modelsUsed:      modelsUsed,
            submitter:       msg.sender,
            timestamp:       block.timestamp,
            blockNumber:     block.number
        });

        hashToRecordIds[fileHash].push(recordId);
        submitterToRecordIds[msg.sender].push(recordId);

        emit DetectionSubmitted(
            recordId,
            fileHash,
            msg.sender,
            isFake,
            fakeProbBasisPts,
            block.timestamp
        );

        return recordId;
    }

    // -----------------------------------------------------------------------
    // Query functions
    // -----------------------------------------------------------------------

    /**
     * @notice Get a single detection record by its ID.
     */
    function getRecord(uint256 recordId)
        external
        view
        returns (DetectionRecord memory)
    {
        require(recordId < totalRecords, "Record does not exist");
        return records[recordId];
    }

    /**
     * @notice Get all record IDs for a given file hash.
     *         Allows checking a file's full detection history.
     */
    function getRecordsByHash(string calldata fileHash)
        external
        view
        returns (uint256[] memory)
    {
        return hashToRecordIds[fileHash];
    }

    /**
     * @notice Get all record IDs submitted by a specific wallet.
     */
    function getRecordsBySubmitter(address submitter)
        external
        view
        returns (uint256[] memory)
    {
        return submitterToRecordIds[submitter];
    }

    /**
     * @notice Get the latest detection record for a file hash.
     *         Returns the most recent analysis result.
     */
    function getLatestRecord(string calldata fileHash)
        external
        view
        returns (DetectionRecord memory, uint256 recordId)
    {
        uint256[] memory ids = hashToRecordIds[fileHash];
        require(ids.length > 0, "No records found for this file hash");
        recordId = ids[ids.length - 1];
        return (records[recordId], recordId);
    }

    /**
     * @notice Check if a file has ever been analyzed.
     */
    function isAnalyzed(string calldata fileHash)
        external
        view
        returns (bool)
    {
        return hashToRecordIds[fileHash].length > 0;
    }

    /**
     * @notice Get detection statistics for a file hash.
     *         Returns counts of fake vs real verdicts across all analyses.
     */
    function getFileStats(string calldata fileHash)
        external
        view
        returns (
            uint256 totalAnalyses,
            uint256 fakeVerdicts,
            uint256 realVerdicts,
            uint256 avgFakeProbBasisPts
        )
    {
        uint256[] memory ids = hashToRecordIds[fileHash];
        totalAnalyses = ids.length;

        if (totalAnalyses == 0) {
            return (0, 0, 0, 0);
        }

        uint256 probSum = 0;
        for (uint256 i = 0; i < ids.length; i++) {
            DetectionRecord memory rec = records[ids[i]];
            if (rec.isFake) {
                fakeVerdicts++;
            } else {
                realVerdicts++;
            }
            probSum += rec.fakeProbability;
        }

        avgFakeProbBasisPts = probSum / totalAnalyses;
    }

    /**
     * @notice Verify a file: returns verdict + timestamp of latest analysis.
     *         Convenience function for the public verification portal.
     */
    function verifyFile(string calldata fileHash)
        external
        view
        returns (
            bool    analyzed,
            bool    latestVerdict,
            uint256 latestTimestamp,
            uint256 analysisCount
        )
    {
        uint256[] memory ids = hashToRecordIds[fileHash];
        if (ids.length == 0) {
            return (false, false, 0, 0);
        }

        DetectionRecord memory latest = records[ids[ids.length - 1]];
        return (
            true,
            latest.isFake,
            latest.timestamp,
            ids.length
        );
    }
}
