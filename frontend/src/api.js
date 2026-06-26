import axios from "axios";

const BASE = import.meta.env.VITE_API_URL || "http://localhost:8000";
const api  = axios.create({ baseURL: BASE });

export async function detectImage(file, threshold = 0.65) {
  const form = new FormData();
  form.append("file", file);
  const { data } = await api.post(
    `/api/detect/image?threshold=${threshold}`,
    form,
    { headers: { "Content-Type": "multipart/form-data" } }
  );
  return data;
}

export async function detectVideo(file, threshold = 0.65) {
  const form = new FormData();
  form.append("file", file);
  const { data } = await api.post(
    `/api/detect/video?threshold=${threshold}`,
    form,
    { headers: { "Content-Type": "multipart/form-data" } }
  );
  return data;
}

export async function pollVideoResult(taskId) {
  const { data } = await api.get(`/api/result/${taskId}`);
  return data;
}

function sleep(ms) { return new Promise(r => setTimeout(r, ms)); }

export async function detectVideoAndWait(file, onProgress, threshold = 0.65) {
  const { task_id } = await detectVideo(file, threshold);
  for (let i = 0; i < 100; i++) {
    await sleep(3000);
    const result = await pollVideoResult(task_id);
    if (onProgress) onProgress(i + 1, 100);
    if (result.status === "complete") return result;
    if (result.status === "failed") throw new Error(result.error || "Video detection failed");
  }
  throw new Error("Timeout waiting for video result");
}

export async function verifyByHash(fileHash) {
  const { data } = await api.get(`/api/verify/${fileHash}`);
  return data;
}

export async function submitToBlockchain(payload) {
  const { data } = await api.post("/api/blockchain/submit", payload);
  return data;
}

export async function computeFileHash(file) {
  const buffer = await file.arrayBuffer();
  const hashBuf = await crypto.subtle.digest("SHA-256", buffer);
  return Array.from(new Uint8Array(hashBuf))
    .map(b => b.toString(16).padStart(2, "0"))
    .join("");
}