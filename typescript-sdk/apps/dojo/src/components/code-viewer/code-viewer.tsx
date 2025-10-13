import { useMemo, useState } from "react";
import { FileTree } from "@/components/file-tree/file-tree";
import { CodeEditor } from "./code-editor";
import { FeatureFile } from "@/types/feature";
import { useURLParams } from "@/contexts/url-params-context";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";

export default function CodeViewer({ codeFiles }: { codeFiles: FeatureFile[] }) {
  const { file, setCodeFile, codeLayout } = useURLParams();

  const selectedFile = useMemo(
    () => codeFiles.find((f) => f.name === file) ?? codeFiles[0],
    [codeFiles, file],
  );

  if (codeLayout === "tabs") {
    return (
      <div className="flex flex-col h-full">
        <Tabs
          value={selectedFile?.name}
          onValueChange={setCodeFile}
          className="flex-1 flex flex-col"
        >
          <TabsList className="w-full justify-start h-auto flex-wrap p-1 gap-1 bg-white dark:bg-cpk-docs-dark-bg rounded-none">
            {codeFiles.map((file) => (
              <TabsTrigger
                key={file.name}
                value={file.name}
                className="border-0 shadow-none text-gray-600 dark:text-neutral-300 hover:bg-foreground/5 hover:text-gray-900 dark:hover:text-neutral-100 data-[state=active]:bg-foreground/8 data-[state=active]:text-gray-900 dark:data-[state=active]:text-white"
              >
                {file.name.split("/").pop()}
              </TabsTrigger>
            ))}
          </TabsList>
          {codeFiles.map((file) => (
            <TabsContent
              key={file.name}
              value={file.name}
              className="flex-1 mt-0 data-[state=inactive]:hidden"
            >
              <div className="h-full bg-gray-50 dark:bg-[#1e1e1e]">
                <CodeEditor file={file} />
              </div>
            </TabsContent>
          ))}
        </Tabs>
      </div>
    );
  }

  return (
    <div className="flex h-full">
      <div className="w-72 border-r border-gray-200 dark:border-neutral-700 flex flex-col bg-white dark:bg-cpk-docs-dark-bg">
        <div className="flex-1 overflow-auto">
          <FileTree files={codeFiles} selectedFile={selectedFile} onFileSelect={setCodeFile} />
        </div>
      </div>
      <div className="flex-1 h-full py-5 bg-gray-50 dark:bg-[#1e1e1e]">
        {selectedFile ? (
          <div className="h-full">
            <CodeEditor file={selectedFile} />
          </div>
        ) : (
          <div className="flex items-center justify-center h-full text-muted-foreground dark:text-neutral-300">
            Select a file to view its content.
          </div>
        )}
      </div>
    </div>
  );
}
