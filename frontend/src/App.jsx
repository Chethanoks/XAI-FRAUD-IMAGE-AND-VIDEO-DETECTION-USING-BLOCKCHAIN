import { BrowserRouter, Routes, Route, NavLink } from "react-router-dom";
import { Toaster } from "react-hot-toast";
import HomePage   from "./pages/HomePage";
import VerifyPage from "./pages/VerifyPage";

function Nav() {
  return (
    <nav style={{
      position: "fixed", top: 0, left: 0, right: 0, zIndex: 100,
      display: "flex", alignItems: "center", justifyContent: "space-between",
      padding: "14px 48px",
      background: "rgba(247,246,250,0.93)",
      backdropFilter: "blur(16px)",
      borderBottom: "1px solid var(--border)",
    }}>
      <div style={{fontFamily:"Syne,sans-serif", fontWeight:800, fontSize:19, color:"var(--ink)"}}>
        🛡️ Deep<span style={{color:"var(--violet)"}}>Guard</span>
      </div>

      <div style={{
        display:"flex", gap:4,
        background:"rgba(15,14,23,0.06)",
        borderRadius:100, padding:4,
      }}>
        {[
          {to:"/",      label:"Home",   end:true},
          {to:"/verify",label:"Verify"},
        ].map(({to, label, end}) => (
          <NavLink key={to} to={to} end={end}
            style={({isActive}) => ({
              fontFamily:"Syne,sans-serif", fontWeight:600, fontSize:13,
              padding:"8px 18px", borderRadius:100, border:"none",
              cursor:"pointer", textDecoration:"none",
              background: isActive ? "var(--violet)" : "transparent",
              color: isActive ? "white" : "var(--ink2)",
              transition:"all 0.2s",
            })}>
            {label}
          </NavLink>
        ))}
      </div>

      <div style={{
        fontFamily:"DM Mono,monospace", fontSize:12,
        color:"var(--violet)", background:"var(--violet-light)",
        border:"1px solid rgba(91,78,232,0.2)",
        padding:"7px 14px", borderRadius:100,
        display:"flex", alignItems:"center", gap:7,
      }}>
        <span style={{width:7,height:7,background:"var(--violet)",borderRadius:"50%",display:"inline-block",animation:"pulse 2s infinite"}} />
        AI + XAI + Blockchain
      </div>

      <style>{`
        @keyframes pulse{0%,100%{opacity:1}50%{opacity:.4}}
      `}</style>
    </nav>
  );
}

export default function App() {
  return (
    <BrowserRouter>
      <div style={{minHeight:"100vh", background:"var(--surface)"}}>
        <Nav />
        <main style={{paddingTop:72}}>
          <Routes>
            <Route path="/"       element={<HomePage />} />
            <Route path="/verify" element={<VerifyPage />} />
          </Routes>
        </main>
        <Toaster position="bottom-right" toastOptions={{
          style:{
            fontFamily:"Syne,sans-serif", fontWeight:600, fontSize:13,
            background:"var(--ink)", color:"white",
            borderRadius:12, padding:"12px 18px",
          }
        }}/>
      </div>
    </BrowserRouter>
  );
}
