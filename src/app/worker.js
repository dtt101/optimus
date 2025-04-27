import { pipeline, env } from "@huggingface/transformers";

env.allowLocalModels = false;

class PipelineSingleton {
  static task = "text-generation";
  static model = "dtt101/ft-gpt2-edu-onnx-int8";
  static instance = null;
  static isGenerating = false;

  static async getInstance(progress_callback = null) {
    if (this.instance === null) {
      try {
        this.instance = pipeline(this.task, this.model, {
          progress_callback,
          quantized: true,
          dtype: "int8",
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

    const prompt = `<QUERY> ${text.trim()} <ANS> `;

    const output = await generator(prompt, {
      do_sample: true,
      top_k: 40,
      top_p: 0.9,
      temperature: 0.7,
      max_new_tokens: 4, // room for two full words
      min_new_tokens: 2, // forces at least 2 tokens → fewer fragments
      repetition_penalty: 1.05,
      stop: ["\n", "<|endoftext|>"],
      return_full_text: false, // we’ll get only the completion
    });

    const fullText = output[0].generated_text.replace(/\n/g, "").trim();

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
