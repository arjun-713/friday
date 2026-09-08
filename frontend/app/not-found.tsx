import Link from "next/link";
import Brand from "../components/Brand";
import { ArrowRightIcon } from "@heroicons/react/24/outline";

export default function NotFound() {
  return <main className="system-page"><Brand/><div><span className="system-code">404</span><h1>This page isn’t here.</h1><p>Head back to Friday, or open your troubleshooting workspace.</p><Link className="system-primary" href="/app">Open workspace <ArrowRightIcon/></Link><Link className="system-secondary" href="/">Back to home</Link></div></main>;
}
