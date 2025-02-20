import { indexName } from "@/config";
import { createPineconeIndex, updatePinecone } from "@/utils";
import { PDFLoader } from "@langchain/community/document_loaders/fs/pdf";
import { Pinecone } from "@pinecone-database/pinecone";
import { DirectoryLoader } from "langchain/document_loaders/fs/directory";
import { TextLoader } from "langchain/document_loaders/fs/text";
import { NextResponse } from "next/server";

export async function POST() {
  try {
    const loader = new DirectoryLoader("./documents", {
      ".txt": (path) => new TextLoader(path),
      ".md": (path) => new TextLoader(path),
      ".pdf": (path) => new PDFLoader(path),
    });

    const docs = await loader.load();
    const vectorDimensions = 1536;

    const client = new Pinecone({
      apiKey: process.env.PINECONE_API_KEY || "",
    });

    await createPineconeIndex(client, indexName, vectorDimensions);
    await updatePinecone(client, indexName, docs);

    return NextResponse.json({
      data: `Successfully created index and loaded data into Pinecone`,
    });
  } catch (error) {
    return NextResponse.json(
      {
        message: "Internal Server Error",
      },
      {
        status: 500,
      },
    );
  }
}
