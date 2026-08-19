const SCARF_PIXEL_ID = "1c040678-b704-471e-a3f5-69c6bf52b703";

export function ScarfPixel() {
  return (
    // Analytics tracking pixel: must stay a plain <img> so the request hits
    // scarf.sh directly; next/image optimization/proxying would break it.
    // eslint-disable-next-line @next/next/no-img-element
    <img
      referrerPolicy="no-referrer-when-downgrade"
      src={`https://static.scarf.sh/a.png?x-pxid=${SCARF_PIXEL_ID}`}
      alt=""
      aria-hidden="true"
      className="absolute top-0 left-0"
    />
  );
}
