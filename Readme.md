# Graph-RAG

This is a Graph RAG pipeline project where the graph will be aware of contradiction between the sources.
This project is split into four phases, where the first is just a simple Vanilla RAG pipeline with Langgraph and chromaDB.

## Topic

`Do larger language models actually reason better, or just pattern-match better?`

## Phase-1 Steps

1. Have a class to fetch documents related to a topic from arXiv api, then chunk them and then store the chunks in a csv file.
   1. So after fetching the documents from arxiv, store the metadata like title, id, pdf_url and abstract in csv file
   2. On the extractor download the pdf, send it to the grobid and then extract the content and add the extracted content to a separate csv file with only one column and update the content csv_file path in the parent csv_file
   3. Folder structure: data -> metadata.csv, extracted_content, extracted_content > [article_id.csv]
2. For chunking will use section chunking and for each document will have a separate csv file.
3. Once the chunking is done will embed them and push them into chromaDB.
   1. along with vector we also need to push metadata
      1. The problem is where can I get the metadata I could again parse the csv file but its not efficient maybe I can the metadata in tag when I create the md file and parse it separately while chunking.
   2. for every chunk we will push the article ID as a metadata field
   3. Add for now in the chromaDB along with the vector will also push actual chunk content
   4. Now using the id we can easily point out to which research paper the answer was referred from but we can't link to where exactly in the research paper. that needs some more tweaking while processing the xml. maybe we can do that later.
4. After will create a dataset for evaluation and test it.
   1. For evaluation from the metadata collection in chromadb pull 3 documents randomly
   2. extract the content from their pdfs and randomly choose a bigger chunk using [i:j], and pass it to an llm along with the papers abstract for gettting the query. then store the query in a {collection}\_test in chromadb
   3. Now we have a chromaDB collection which has an query, document, article_id and embedding for that query using this we can easily run an eval script
   4. the test collections already has an big chunk will check if the smaller chunk the collection.query returns is in the big chunk
5. Once the evaluation is done will connect LLM to the graph
   1. reorganize the files and built the graph

## Tasks left

1. RAG is done, now have to figure out how to deploy this. first I need to figure out how to deploy the backend
2. In the backend there is the langgraph server, I don't know how authentication between langgraph server and langgraph api works
3. Then there is local embedding model which either I can use hugging face or VoyageAI. HuggingFace would be better cause It has a langchain helper so I can just reuse the client with small change
4. The main problem is the grobbid server. maybe they have a public server if there is its great or else I don't know
5. For chroma I can just create an free account in chroma cloud and use that.

## Commands

1. docker run --rm --init --ulimit core=0 -p 8070:8070 grobid/grobid:0.9.0-crf ( To start the grobid cpu only lightweight version)

## Todo

After this I will remove the grobID and change the embedding model to an hosted one like voyage AI

## Objective

So I have decided not to host the pipeline building service. I will just maybe turn that into an docker file and add instructions with docker-compose on how to build their own source.

The thing that will be actually deployed are the langgraph server through langgraph cloud and then an next.js frontend that communicates with the langgraph server and all the vector will be stored in chroma cloud

## Things to do now:

1. Change the ChromaDB persistent client to chroma cloud
2. the embedding client should have a fallback to use the bge model provided by chromaDB if the embedding server is not reachable.
