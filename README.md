# DeepGuard — Deepfake Detection with XAI + Blockchain

> AI-powered deepfake detection for images and videos, with multi-layer explainability
> and an immutable blockchain audit trail on Polygon.

---

## What makes this different

| Feature | Most tools | DeepGuard |
|---|---|---|
| Multi-model ensemble | ❌ | ✅ CLIP + AltFreezing + LipForensics |
| XAI heatmaps | Basic / none | ✅ Attention Rollout + SHAP + Artifact Classifier |
| Artifact type labeling | ❌ | ✅ Face-swap / GAN / Lip-sync / Texture |
| Blockchain audit trail | ❌ | ✅ Polygon — immutable, public |
| Public hash verification | ❌ | ✅ Anyone can verify any file by hash |
| Video temporal analysis | Partial | ✅ Optical flow + frame timeline |

---

## Architecture

```
Image  →  CLIP fine-tuned (SBI strategy)  →  XAI pipeline
Video  →  AltFreezing + LipForensics + Temporal Consistency  →  Weighted ensemble

XAI:
  Layer 1: Attention Rollout   (which facial patches)
  Layer 2: SHAP pixel values   (per-pixel contribution)
  Layer 3: Artifact Classifier (what type of manipulation)

Blockchain:
  SHA-256 hash + verdict + confidence → Solidity contract → Polygon Mumbai
  Public verification portal: paste any hash → see full history
```

---

## Project Structure

```
deepfake-detector/
├── ai/
│   ├── models/
│   │   ├── clip_image_detector.py   # CLIP + SBI image detection
│   │   ├── altfreezing_video.py     # Spatial+temporal video detection
│   │   ├── lipforensics.py          # Lip-sync detection
│   │   └── video_ensemble.py        # Weighted ensemble of all video models
│   ├── xai/
│   │   └── explainer.py             # Attention Rollout + SHAP + Artifact Classifier
│   └── preprocessing/
│       ├── face_detector.py         # MTCNN face detection + alignment
│       └── frame_extractor.py       # Adaptive frame extraction + optical flow
├── backend/
│   ├── main.py                      # FastAPI routes
│   └── blockchain.py                # Web3 contract interaction
├── contracts/
│   └── DeepfakeAudit.sol            # Solidity smart contract
├── frontend/
│   └── src/
│       ├── pages/
│       │   ├── HomePage.jsx         # Upload + detect interface
│       │   └── VerifyPage.jsx       # Public hash verification portal
│       ├── components/
│       │   ├── Uploader.jsx         # Drag-and-drop file upload
│       │   └── ResultCard.jsx       # Detection result + XAI + blockchain
│       ├── hooks/
│       │   └── useBlockchain.js     # MetaMask + contract hook
│       └── api.js                   # API service layer
├── requirements.txt
└── .env.example
```

---

## Setup

### 1. Clone and install Python deps

```bash
git clone https://github.com/Chethanoks/deepguard
cd deepfake-detector

# Use Python 3.11 (required for torch + tensorflow compat)
python3.11 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Set environment variables

```bash
cp .env.example .env
# Edit .env with your values
```

### 3. Run the backend

```bash
cd deepfake-detector
uvicorn backend.main:app --reload --port 8000
```

### 4. Run the frontend

```bash
cd frontend
npm install
npm run dev
# Open http://localhost:3000
```

---

## Blockchain Setup

### Deploy the smart contract

1. Install Hardhat or Remix IDE
2. Deploy `contracts/DeepfakeAudit.sol` to Polygon Mumbai testnet
3. Copy the deployed contract address to `.env` → `CONTRACT_ADDRESS`
4. Get free Mumbai MATIC from [faucet.polygon.technology](https://faucet.polygon.technology)
5. Add your MetaMask private key to `.env` → `WALLET_PRIVATE_KEY`

### Get a free RPC URL

1. Go to [alchemy.com](https://www.alchemy.com)
2. Create an app → select Polygon Mumbai
3. Copy the HTTPS URL to `.env` → `POLYGON_RPC_URL`

---


## Dataset
Trained on 140k Real and Fake Faces (Kaggle) + custom photos.
EfficientNet-B4 fine-tuned for 15 epochs on RTX 3050 GPU.
Accuracy: 92-95% on test set.
Blockchain: Deployed on Polygon Amoy Testnet (chainId 80002)


## Training the Models

### Image model (CLIP + SBI)

```bash
python -m ai.models.clip_image_detector train \
  --dataset path/to/FaceForensics++ \
  --epochs 20 \
  --output checkpoints/clip_detector.pt
```

### Video model (AltFreezing)

```bash
python -m ai.models.altfreezing_video train \
  --dataset path/to/DFDC \
  --epochs 30 \
  --output checkpoints/altfreezing.pt
```

### Recommended datasets

| Dataset | Download |
|---|---|
| FaceForensics++ | [github.com/ondyari/FaceForensics](https://github.com/ondyari/FaceForensics) |
| DFDC | [ai.facebook.com/datasets/dfdc](https://ai.facebook.com/datasets/dfdc) |
| Celeb-DF v2 | [github.com/yuezunli/celeb-deepfakeforensics](https://github.com/yuezunli/celeb-deepfakeforensics) |

---

## API Endpoints

| Method | Route | Description |
|---|---|---|
| POST | `/api/detect/image` | Analyze image (multipart/form-data) |
| POST | `/api/detect/video` | Queue video for analysis |
| GET  | `/api/result/{task_id}` | Poll async video result |
| GET  | `/api/verify/{hash}` | Check blockchain history by SHA-256 hash |
| POST | `/api/blockchain/submit` | Submit result to blockchain |
| GET  | `/api/health` | Health check |

---

## Tech Stack

```
AI/ML       PyTorch, HuggingFace, OpenAI CLIP, facenet-pytorch, OpenCV
XAI         SHAP, captum (Attention Rollout), custom Artifact Classifier
Backend     FastAPI, Uvicorn, Web3.py
Blockchain  Solidity, Polygon Mumbai, Ethers.js
Frontend    React, Vite, TailwindCSS, Recharts, react-dropzone
```

---

Built by Chethan K S
