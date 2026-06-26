import { useCallback, useState } from "react";
import { useDropzone } from "react-dropzone";

const ACCEPTED = {
  "image/jpeg":[".jpg",".jpeg"],"image/png":[".png"],"image/webp":[".webp"],
  "video/mp4":[".mp4"],"video/avi":[".avi"],"video/webm":[".webm"],"video/quicktime":[".mov"],
};

export default function Uploader({ onAnalyze, analyzing }) {
  const [file, setFile]       = useState(null);
  const [preview, setPreview] = useState(null);
  const [error, setError]     = useState(null);

  const onDrop = useCallback((accepted, rejected) => {
    setError(null);
    if (rejected.length > 0) { setError(rejected[0].errors[0]?.message || "Invalid file"); return; }
    const f = accepted[0];
    setFile(f);
    setPreview(f.type.startsWith("image/") ? URL.createObjectURL(f) : null);
  }, []);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop, accept: ACCEPTED, maxFiles: 1, maxSize: 200*1024*1024, disabled: analyzing,
  });

  const clear = (e) => { e.stopPropagation(); setFile(null); setPreview(null); setError(null); };

  return (
    <div>
      <div {...getRootProps()} style={{
        border: `2px dashed ${isDragActive ? "var(--violet)" : "var(--border2)"}`,
        borderRadius: 16,
        padding: preview ? 20 : 52,
        textAlign: "center",
        cursor: analyzing ? "not-allowed" : "pointer",
        background: isDragActive ? "var(--violet-light)" : "var(--white)",
        transition: "all 0.2s",
        opacity: analyzing ? 0.6 : 1,
      }}>
        <input {...getInputProps()} />

        {preview ? (
          <div style={{position:"relative"}}>
            <button onClick={clear} style={{
              position:"absolute", top:-8, right:-8, zIndex:10,
              width:26, height:26, borderRadius:"50%",
              background:"var(--red)", color:"white", border:"none",
              cursor:"pointer", fontSize:14, lineHeight:1,
            }}>✕</button>
            <img src={preview} alt="preview"
              style={{maxHeight:200, borderRadius:12, objectFit:"contain", margin:"0 auto", display:"block"}} />
            <div style={{
              marginTop:12, display:"flex", alignItems:"center", justifyContent:"center",
              gap:8, fontFamily:"DM Mono,monospace", fontSize:12, color:"var(--muted)",
            }}>
              <span style={{width:6,height:6,borderRadius:"50%",background:"var(--violet)",display:"inline-block"}}/>
              {file.name} · {(file.size/1024/1024).toFixed(1)} MB
            </div>
          </div>
        ) : (
          <div>
            <div style={{
              width:60, height:60, borderRadius:16,
              background:"var(--violet-light)",
              border:"1px solid rgba(91,78,232,0.15)",
              display:"flex", alignItems:"center", justifyContent:"center",
              fontSize:28, margin:"0 auto 16px",
            }}>🔍</div>
            <div style={{fontFamily:"Syne,sans-serif", fontWeight:700, fontSize:17, marginBottom:6}}>
              {isDragActive ? "Drop it here" : "Drop image or video"}
            </div>
            <div style={{color:"var(--muted)", fontSize:13, marginBottom:16}}>or click to browse files</div>
            <div style={{display:"flex", flexWrap:"wrap", gap:6, justifyContent:"center"}}>
              {["JPG","PNG","WebP","MP4","AVI","MOV"].map(f => (
                <span key={f} style={{
                  fontFamily:"DM Mono,monospace", fontSize:11,
                  padding:"3px 10px", borderRadius:100,
                  background:"var(--violet-light)", color:"var(--violet)",
                  border:"1px solid rgba(91,78,232,0.2)",
                }}>{f}</span>
              ))}
            </div>
          </div>
        )}
      </div>

      {error && <p style={{color:"var(--red)", fontSize:13, marginTop:8, textAlign:"center"}}>{error}</p>}

      {file && !analyzing && (
        <button onClick={() => onAnalyze(file)} style={{
          width:"100%", marginTop:14, padding:"14px",
          fontFamily:"Syne,sans-serif", fontWeight:700, fontSize:15,
          background:"var(--violet)", color:"white",
          border:"none", borderRadius:100,
          cursor:"pointer", transition:"all 0.2s",
          display:"flex", alignItems:"center", justifyContent:"center", gap:8,
        }}
        onMouseEnter={e=>e.target.style.background="var(--ink)"}
        onMouseLeave={e=>e.target.style.background="var(--violet)"}
        >
          🔬 Analyze {file.type.startsWith("image/") ? "Image" : "Video"}
        </button>
      )}

      {analyzing && (
        <div style={{
          marginTop:14, padding:"14px",
          background:"var(--violet-light)",
          border:"1px solid rgba(91,78,232,0.2)",
          borderRadius:100, textAlign:"center",
          fontFamily:"Syne,sans-serif", fontWeight:600, fontSize:14,
          color:"var(--violet)", display:"flex", alignItems:"center", justifyContent:"center", gap:10,
        }}>
          <span style={{
            width:16, height:16, border:"2px solid rgba(91,78,232,0.3)",
            borderTopColor:"var(--violet)", borderRadius:"50%",
            display:"inline-block", animation:"spin 0.8s linear infinite",
          }}/>
          Analyzing with AI...
          <style>{`@keyframes spin{to{transform:rotate(360deg)}}`}</style>
        </div>
      )}
    </div>
  );
}
