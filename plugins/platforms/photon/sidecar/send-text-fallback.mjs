// Helpers for Photon iMessage text sends.
//
// spectrum-ts' markdown builder adds iMessage data-detection metadata when
// markdown contains links. Some Photon IMAgentKit send paths reject that option
// with `enable_data_detection is not supported`. Retry those specific failures
// as plain text so Hermes doesn't silently lose the whole response.

export function isUnsupportedDataDetectionError(error) {
  const text = error && error.stack ? error.stack : String(error);
  return (
    text.includes("enable_data_detection is not supported") ||
    text.includes("enableDataDetection is not supported")
  );
}

export async function sendTextWithMarkdownFallback(
  space,
  text,
  format,
  { spectrumText, spectrumMarkdown, log = console.error }
) {
  if (format !== "markdown") {
    return await space.send(spectrumText(text));
  }

  try {
    return await space.send(spectrumMarkdown(text));
  } catch (error) {
    if (!isUnsupportedDataDetectionError(error)) {
      throw error;
    }
    log(
      "photon-sidecar: markdown send rejected enable_data_detection; " +
        "retrying as plain text"
    );
    return await space.send(spectrumText(text));
  }
}
