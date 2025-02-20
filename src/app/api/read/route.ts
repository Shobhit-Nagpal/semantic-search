import { indexName } from "@/config";
import { queryPineconeVectorStoreAndQueryLLM } from "@/utils";
import { Pinecone } from "@pinecone-database/pinecone";
import { NextRequest, NextResponse } from "next/server";

export async function POST(req: NextRequest) {
  try {
    const body = await req.json();
    const client = new Pinecone({
      apiKey: process.env.PINECONE_API_KEY || "",
    });

    const text = await queryPineconeVectorStoreAndQueryLLM(
      client,
      indexName,
      body,
    );

    return NextResponse.json({
      data: text,
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
