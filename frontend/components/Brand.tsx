import Link from "next/link";

export function BrandMark({ className = "" }: { className?: string }) {
  return <svg className={`brand-symbol ${className}`} viewBox="0 0 32 32" fill="none" aria-hidden="true"><path d="M7 25V9a3 3 0 0 1 3-3h15M7 16h14" stroke="currentColor" strokeWidth="4" strokeLinecap="round"/><circle cx="25" cy="25" r="3" fill="currentColor"/></svg>;
}

export default function Brand() {
  return <Link className="brand" href="/" aria-label="Friday home"><BrandMark/><span>friday<span className="brand-period">.</span></span></Link>;
}
