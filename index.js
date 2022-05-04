import { useEffect, useRef, useState } from "react";
import * as toxicity from "@tensorflow-models/toxicity";

export default async function textToxicity(text) {
  const model = useRef();

  if (!text) return;
  model.current = model.current || (await toxicity.load());
  const result = await model.current.classify([text]).catch(() => {});

  if (!result) return;

  return result.map((prediction) => {
    const [{ match, probabilities }] = prediction.results;
    return {
      label: prediction.label,
      match,
      text,
      probabilities,
      probability: (probabilities[1] * 100).toFixed(2) + "%",
    };
  });
}
