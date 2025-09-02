import * as dotenv from "dotenv";
dotenv.config();
import { Pinecone } from "@pinecone-database/pinecone";
import { GoogleGenerativeAIEmbeddings } from "@langchain/google-genai";

// configure pinecode DB
const pinecone = new Pinecone();
const pineconeIndex = pinecone.Index(process.env.PINECONE_INDEX_NAME);

// configure Google generativeAI embeddings
const googleGenAIEmbeddings = new GoogleGenerativeAIEmbeddings({
  apiKey: process.env.GEMINI_API_KEY,
  model: "text-embedding-004",
});

module.exports = {
  pineconeIndex,
  googleGenAIEmbeddings,
};
