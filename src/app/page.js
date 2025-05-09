"use client";

import { useState, useEffect, useRef, useCallback } from "react";

export default function Home() {
  const [result, setResult] = useState(null);
  const [ready, setReady] = useState(null);

  const worker = useRef(null);

  useEffect(() => {
    if (!worker.current) {
      worker.current = new Worker(new URL("./worker.js", import.meta.url), {
        type: "module",
      });
    }

    const onMessageReceived = (e) => {
      switch (e.data.status) {
        case "initiate":
          setReady(false);
          break;
        case "ready":
          setReady(true);
          break;
        case "complete":
          setResult(e.data.output);
          break;
      }
    };

    worker.current.addEventListener("message", onMessageReceived);

    return () =>
      worker.current.removeEventListener("message", onMessageReceived);
  });

  const classify = useCallback((text) => {
    if (worker.current) {
      worker.current.postMessage({ text });
    }
  }, []);

  return (
    <main className="flex min-h-screen flex-col items-center justify-center p-12">
      <h1 className="text-5xl font-bold mb-2 text-center">Optimus</h1>
      <h2 className="text-2xl mb-4 text-center">
        Client-side search autocomplete
      </h2>

      <input
        className="w-md max-w-xs p-2 border border-gray-300 rounded mb-4"
        type="text"
        placeholder="Enter text here"
        onInput={(e) => {
          classify(e.target.value);
        }}
      />

      {ready !== null && (
        <div
          className="bg-black-100 p-2 rounded"
          style={{
            maxWidth: "80%",
            overflowX: "auto",
            fontFamily: "monospace",
          }}
        >
          {!ready || !result ? "Loading..." : JSON.stringify(result, null, 2)}
        </div>
      )}
    </main>
  );
}
