import { pipeline, env } from "@huggingface/transformers";

env.allowLocalModels = false;

class PipelineSingleton {
  static task = "text-generation";
  static model = "Xenova/distilgpt2";
  static instance = null;
  static isGenerating = false;

  static async getInstance(progress_callback = null) {
    if (this.instance === null) {
      try {
        this.instance = pipeline(this.task, this.model, {
          progress_callback,
          dtype: "fp32",
        });
      } catch (error) {
        console.error("Error initializing model:", error);
        throw error;
      }
    }
    return this.instance;
  }
}

let requestQueue = [];
let isProcessing = false;

async function processQueue() {
  if (isProcessing || requestQueue.length === 0) return;

  isProcessing = true;
  const { text, postMessage } = requestQueue.shift();

  try {
    const generator = await PipelineSingleton.getInstance((x) => {
      postMessage(x);
    });

    const context = `Complete the British‑curriculum TEACHER search query with ONE or TWO key words.\n
      +  Partial query: `;

    const output = await generator(context + text, {
      temperature: 0.2,
      max_new_tokens: 4,
      top_k: 40,
      top_p: 0.9,
      repetition_penalty: 1.1,
      stop_sequences: ["\n"],
    });

    const fullText = output[0].generated_text
      .substring(context.length)
      .replace(/\n/g, "")
      .trim();

    postMessage({
      status: "complete",
      output: fullText,
    });
  } catch (error) {
    console.error("Error generating text:", error);
    postMessage({
      status: "error",
      error: error.message,
      output: "",
    });
  } finally {
    isProcessing = false;
    setTimeout(() => processQueue(), 50);
  }
}

self.addEventListener("message", (event) => {
  requestQueue.push({
    text: event.data.text,
    postMessage: (message) => self.postMessage(message),
  });

  if (!isProcessing) {
    processQueue();
  }
});
