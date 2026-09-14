/** Declare the encoding SSE already requires so response observers decode it correctly. */
export function withUtf8SseResponse(response: Response) {
  const contentType = response.headers.get("content-type");
  if (
    contentType?.split(";", 1)[0].trim().toLowerCase() !==
      "text/event-stream" ||
    /;\s*charset\s*=/i.test(contentType)
  ) {
    return response;
  }
  const headers = new Headers(response.headers);
  headers.set("content-type", `${contentType}; charset=utf-8`);
  return new Response(response.body, {
    status: response.status,
    statusText: response.statusText,
    headers,
  });
}
