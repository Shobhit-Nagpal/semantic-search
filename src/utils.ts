import { OpenAI, OpenAIEmbeddings } from "@langchain/openai";
import { loadQAStuffChain } from "langchain/chains";
import { Document } from "langchain/document";
import { RecursiveCharacterTextSplitter } from "langchain/text_splitter";
import { timeout } from "./config";
import { Pinecone } from "@pinecone-database/pinecone";
import { Vector } from "./types";

export const createPineconeIndex = async (
  client: Pinecone,
  indexName: string,
  vectorDimension: number,
) => {
  console.log(`Checking ${indexName}...`);

  try {
    const existingIndexes = await client.listIndexes();

    if (!existingIndexes.indexes) throw new Error("Indexes is not defined");

    const existingIndexNames = existingIndexes.indexes.map(
      (index) => index.name,
    );

    if (!existingIndexNames.includes(indexName)) {
      console.log(`Creating ${indexName}...`);

      await client.createIndex({
        name: indexName,
        dimension: vectorDimension,
        metric: "cosine",
        spec: {
          serverless: {
            cloud: "aws",
            region: "us-east-1",
          },
        },
      });

      console.log(
        `Creating ${indexName}... please wait for it to finish initializing.`,
      );

      await new Promise((resolve) => setTimeout(resolve, timeout));
    } else {
      console.log(`Index ${indexName} already exists.`);
    }
  } catch (error) {
    console.error(`Error creating Pinecone index: `, error);
  }
};

export const updatePinecone = async (
  client: Pinecone,
  indexName: string,
  documents: Document[],
) => {
  try {
    const index = client.Index(indexName);
    console.log(`Pinecone index retrieved: ${indexName}`);

    for (const doc of documents) {
      console.log(`Processing document: ${doc.metadata.source}`);
      const txtPath = doc.metadata.source;
      const text = doc.pageContent;

      const textSplitter = new RecursiveCharacterTextSplitter({
        chunkSize: 1000,
      });

      console.log(`Splitting text into chunks...`);

      const chunks = await textSplitter.createDocuments([text]);

      console.log(`Text split into ${chunks.length} chunks`);
      console.log(`Converting text chunks into vector embeddings....`);

      // Creates vector representation of a piece of text
      const embeddings = await new OpenAIEmbeddings().embedDocuments(
        chunks.map((chunk) => chunk.pageContent.replace(/\n/g, " ")),
      );

      const batchSize = 100;
      let batch: Vector[] = [];

      for (let i = 0; i < chunks.length; i++) {
        const chunk = chunks[i];

        const vector = {
          id: `${txtPath}_${i}`,
          values: embeddings[i],
          metadata: {
            ...chunk.metadata,
            loc: JSON.stringify(chunk.metadata.loc),
            pageContent: chunk.pageContent,
            txtPath: txtPath,
          },
        };

        batch = [...batch, vector];

        if (batch.length === batchSize || i === chunks.length - 1) {
          await index.upsert(batch);
          batch = [];
        }
      }
    }
  } catch (error) {
    console.error(`Error updating Pinecone: `, error);
  }
};

export const queryPineconeVectorStoreAndQueryLLM = async (
  client: Pinecone,
  indexName: string,
  query: string,
) => {
  try {
    console.log(`Querying Pinecone vector store...`);
    const index = client.Index(indexName);

    // Create query embedding
    const queryEmbedding = await new OpenAIEmbeddings().embedQuery(query);

    // Query Pinecone and return top 10 matches
    const queryResponse = await index.query({
      topK: 10,
      vector: queryEmbedding,
      includeMetadata: true,
      includeValues: true,
    });

    console.log(`Found ${queryResponse.matches.length} matches...`);
    console.log(`Asking question: ${query}`);

    if (queryResponse.matches.length) {
      const llm = new OpenAI({});
      const chain = loadQAStuffChain(llm);

      const concatenatedPageContent = queryResponse.matches
        .map((match) => match.metadata?.pageContent)
        .join(" ");

      const result = await chain.invoke({
        input_documents: [
          new Document({ pageContent: concatenatedPageContent }),
        ],
        question: query,
      });

      console.log(`Result: ${result}`);
      console.log(`Answer: ${result.text}`);

      return result.text;
    } else {
      console.log(`No matches. Not querying the model`);
    }
  } catch (error) {
    console.error(`Error querying Pinecone: `, error);
  }
};
