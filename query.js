import * as dotenv from "dotenv";
dotenv.config();
import { GoogleGenerativeAIEmbeddings } from "@langchain/google-genai";
import readlineSync from "readline-sync";
import { GoogleGenAI } from "@google/genai";
import pineconeIndex from "./config/pincone.config";

const ai = new GoogleGenAI({});
const History = [];

// cofigure embeddings
const embeddings = new GoogleGenerativeAIEmbeddings({
  apiKey: process.env.GEMINI_API_KEY,
  model: "text-embedding-004",
});

// combines the previous query history in order to provide meaningful query
async function transformQuery(prompt) {
  History.push({
    role: "user",
    parts: [{ text: prompt }],
  });

  const response = await ai.models.generateContent({
    model: "gemini-2.0-flash",
    contents: History,
    config: {
      systemInstruction: `You are a query rewriting expert. Based on the provided chat history, rephrase the "Follow Up user Question" into a complete, standalone question that can be understood without the chat history.
        Only output the rewritten question and nothing else.
      `,
    },
  });

  History.pop();

  return response.text;
}

async function chatting(prompt) {
  // convert the question into meaning question based on past conversation
  const queries = await transformQuery(prompt);
  console.log("Query = ", queries);

  // create vector embeddings
  const queryVector = await embeddings.embedQuery(queries);

  // search vector in the database
  const searchResults = await pineconeIndex.query({
    topK: 10,
    vector: queryVector,
    includeMetadata: true,
  });

  // top 10 documents: 10 metadata text part 10 documents
  const context = searchResults.matches
    .map((match) => match.metadata.text)
    .join("\n\n---\n\n");

  // push user prompt history
  History.push({
    role: "user",
    parts: [{ text: queries }],
  });

  const response = await ai.models.generateContent({
    model: "gemini-2.0-flash",
    contents: History,
    config: {
      systemInstruction: `You have to behave like a Data Structure & Algorithm Expert. 
            You will be given a context of a relevant information and a user question.
            Your task is to answer the user's question based ONLY on the provided context.
            If the answer is not in the context, you must say "I could not find the answer in the provided document"
            Keep your answers clear, concise, and educational.

                Context: ${context}
            `,
    },
  });

  // push response history
  History.push({
    role: "model",
    parts: [{ text: response.text }],
  });

  console.log("\n");
  console.log(response.text);
}

async function main() {
  const userProblem = readlineSync.question("Ask me anything---> ");
  await chatting(userProblem);
  main();
}

main();
