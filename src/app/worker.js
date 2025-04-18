import { pipeline, env } from "@huggingface/transformers";

env.allowLocalModels = false;

class PipelineSingleton {
  static task = "text-generation";
  static model = "Xenova/distilgpt2";
  static instance = null;

  static async getInstance(progress_callback = null) {
    if (this.instance === null) {
      this.instance = pipeline(this.task, this.model, {
        progress_callback,
        dtype: "fp32",
      });
    }
    return this.instance;
  }
}

self.addEventListener("message", async (event) => {
  let generator = await PipelineSingleton.getInstance((x) => {
    self.postMessage(x);
  });

  const context =
    "suggest one or two words to complete a search term for an educational search engine for teachers, looking for resources: ";
  let output = await generator(context + event.data.text, {
    temperature: 0.4,
    max_new_tokens: 3,
    repetition_penalty: 1.2,
    no_repeat_ngram_size: 2,
  });

  let fullText = output[0].generated_text
    .substring(context.length)
    .replace(/\n/g, "")
    .trim();
  self.postMessage({
    status: "complete",
    output: fullText,
  });
});
