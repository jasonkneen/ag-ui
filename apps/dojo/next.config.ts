import type { NextConfig } from "next";
import createMDX from "@next/mdx";
import path from "path";

const withMDX = createMDX({
  extension: /\.mdx?$/,
  options: {
    // If you use remark-gfm, you'll need to use next.config.mjs
    // as the package is ESM only
    // https://github.com/remarkjs/remark-gfm#install
    remarkPlugins: [],
    rehypePlugins: [],
    // If you use `MDXProvider`, uncomment the following line.
    providerImportSource: "@mdx-js/react",
  },
});

const nextConfig: NextConfig = {
  /* config options here */
  // Configure pageExtensions to include md and mdx
  pageExtensions: ["ts", "tsx", "js", "jsx", "md", "mdx"],
  webpack: (config, { isServer }) => {
    // Ignore the demo files during build
    config.module.rules.push({
      test: /agent\/demo\/crew_enterprise\/ui\/.*\.(ts|tsx|js|jsx)$/,
      loader: "ignore-loader",
    });

    // Force all @ag-ui/client imports (including those inside CopilotKit's
    // pre-bundled code) to resolve to the local workspace version.
    config.resolve.alias = config.resolve.alias || {};
    config.resolve.alias["@ag-ui/client"] = path.resolve(
      __dirname,
      "../../sdks/typescript/packages/client",
    );

    return config;
  },
  serverExternalPackages: ["@mastra/libsql", "@copilotkit/runtime"],
};

// Merge MDX config with Next.js config
export default withMDX(nextConfig);
