import { BrandMark } from "../../components/Brand";

export default function LoadingWorkspace() {
  return <main className="workspace-loading" role="status"><BrandMark/><p>Opening your workspace…</p></main>;
}
