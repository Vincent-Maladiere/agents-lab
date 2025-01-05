"""
=========
RAG Agent
=========

This notebook takes inspiration from Pydantic-AI `RAG tutorial <https://ai.pydantic.dev/examples/rag/>`_
and simplifies its approach.

Context
-------

Our objective is to create an assistant that answers questions about Logfire
(developed by Pydantic) using the project's documentation as a knowledge base. 
We make this RAG agentic by providing a `retrieve` function as a tool for the LLM.

.. mermaid::

   flowchart TD
      A(User) -->|Ask a question| B(Agent)
      B --> |Messages + retrieve tool|C{LLM}
      C --> |Text or Tool Calling response|B
      B --> |Vectorize user query |D(Knowledge base)
      D --> |Return documents closest to query|B
      B --> |Return answer|A


In this system, the LLM has access to the "retrieve" tool, which it may or may not
invoke in its response. If invoked, the tool call is parsed by the LLM client
and returned as a structured response to the agent, which executes the requested
function.

This differs from **workflow-based RAG**, where the retrieved function is always
executed before calling the LLM. In workflow-based RAG, the LLM prompt is a
concatenation of the initial prompt and the retrieved content.
For more detailed information, we recommend exploring
`Ragger-Duck <https://probabl-ai.github.io/sklearn-ragger-duck/user_guide/index.html>`_,
a RAG implementation developed by scikit-learn core developers.

.. figure:: ../_static/rag_workflow.png

   A RAG workflow (source: Ragger-Duck)

   
RAG Workflow vs Agent-Based RAG
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

We revisit the "workflow vs agent-based systems" tradeoff mentioned in the previous
notebook. This assistant use-case requires a high degree of flexibility, as user
queries are arbitrary. To reduce costs and latency, we only query the knowledge base
when necessary based on user input.
   
Implementation
--------------

Design choices
~~~~~~~~~~~~~~

We simplify the original pydantic tutorial with the following optimizations:

- **No vector database**:

  - For small knowledge bases, overall latency can be reduced by keeping the data
    in memory using a dataframe or a numpy array, persisted with `diskcache <https://grantjenks.com/docs/diskcache/>`_.
  - Approximate nearest neighbors (ANN) operations, typically provided by a vector
    database, can be replaced by a simple Nearest Neighbors estimator from scikit-learn.

- **Batch vectorization**:
  
  - The content of the knowledge base is vectorized in a single batch rather
    than looping through each element individually.

- **Local vectorization**:

  - To reduce API costs, we vectorize content locally by downloading a text vectorizer
    from HuggingFace. This is achieved using ``skrub``, which wraps the
    ``sentence-transformers`` library to provide a scikit-learn-compatible transformer.
    No GPU is required.

Overall, this approach is faster to execute while remaining scalable
for reasonably sized knowledge bases, making it more efficient than the original
Pydantic tutorial.
"""
# %%
#
# Building the knowledge base
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# We begin by fetching the Logfire documentation archived online. Next, we use skrub's
# dataframe visualization to display long text more efficiently (click on any cell
# to view its text content).
import pandas as pd
import requests
from skrub import patch_display


# Replace pandas' dataframe visualisation with skrub's
patch_display()

DOCS_JSON = (
    'https://gist.githubusercontent.com/'
    'samuelcolvin/4b5bb9bb163b1122ff17e29e48c10992/raw/'
    '80c5925c42f1442c24963aaf5eb1a324d47afe95/logfire_docs.json'
)

doc_pages = pd.DataFrame(
    requests.get(DOCS_JSON).json()
)
doc_pages

# %%
# We now use a text embedding model to get a vectorized representation of each page
# of our knowledge base.
# We can choose among various type of vectorizer:
# 
# .. list-table::
#    :header-rows: 1
# 
#    * - Type
#      - Example
#      - Advantages
#      - Caveats 
#    * - Ngram-based
#      - BM25, TF-IDF, LSA, MinHash
#      - Fast and cheap to train. Good baselines.
#      - Lack flexibility, corpus dependent.
#    * - Pre-trained text encoder, open weight
#      - BERT, e5-v2, any model on sentence-transformers
#      - More powerful embedding representations, local inference.
#      - Requires installing pytorch and extra dependencies.
#    * - Pre-trained text encoder, commercial API
#      - open-ai text-embedding-3-small
#      - Most powerful representations, using techniques like
#        `Matryoshka representation learning <https://arxiv.org/abs/2205.13147>`_
#        (also available on sentence-transformers). Easy API integration.
#      - Inference costs, reliance on a third party, closed weights, batch size < 2048.
#      
# For this example, we choose the second option, as it reduces inference cost.
# skrub's TextEncoder downloads the specified model locally using sentence-transformers,
# transformers and pytorch before generating the embeddings for the knowledge base.

from skrub import TextEncoder


text_encoder = TextEncoder(
    model_name="sentence-transformers/paraphrase-albert-small-v2",
    n_components=None,
)
embeddings = text_encoder.fit_transform(doc_pages["content"])
embeddings.shape

# %%
# Next, we use scikit-learn ``NearestNeighbors`` to perform exact retrieval. Note that
# this operation has a time complexity of :math:`O(d \times N)`, where :math:`d` is
# the dimensionality of our embedding vectors, and :math:`N` the number of elements
# to scan. For larger knowledge bases, using Approximate Nearest Neighbors, with
# techniques like HNSW (implemented by `faiss <https://faiss.ai/>`_) or random
# projections (implemented by `Annoy <https://github.com/spotify/annoy>`_)
# is recommended, as these reduce the retrieval time complexity to
# :math:`O(d \times log(N))`.
#
# We return the indices of the 8 closest match and their distances for two queries:
# one related query to our knowledge base topic (Logfire) and another one unrelated
# (cooking).

from sklearn.neighbors import NearestNeighbors


nn = NearestNeighbors(n_neighbors=8).fit(embeddings)

query_embedding = text_encoder.transform(
    pd.Series([
        "How do I configure logfire to work with FastAPI?",
        "I'm a chef, explain how to bake Lasagnas.",
    ])
)

distances, indices = nn.kneighbors(query_embedding, return_distance=True)

print(distances[0])
doc_pages.iloc[indices[0]]

# %%
# We observe that we successfully retrieved content related to FastAPI in the
# documentation. What are the results for the unrelated query?

print(distances[1])
doc_pages.iloc[indices[1]]

# %%
# For the second query, we can hardly discern a link between the retrieved items and
# the original question. However, notice that their distances are higher compared
# to the first query. This means that no article closely match the second query.
#
# The average distances between the first and second queries are quite similar, though.
# This issue is commonly referred as the **curse of dimensionality**, where items in
# high-dimensional spaces tends to all appear "far" from each other due to
# the hyper-volume growing exponentially with the number of dimensions. Real-world
# implementations require a careful evaluation of retrieval system performance, which
# we skip here.
#
# A possible filtering method would be to set a radius, i.e., a maximum distance beyond
# which retrieved elements are discarded. As shown below, the second query results in
# an empty set, as all euclidean distances exceed 14.
nn.radius_neighbors(
    query_embedding,
    radius=14,
    return_distance=True,
    sort_results=True,
)

# %%
# We can emulate persistence on disk using diskcache. Originally designed as a
# fast key-value storage solution for Django, it can also be applied in our
# agentic context. Here, we serialize the knowledge base content, our text encoder,
# and the fitted nearest neighbors estimator.
import diskcache as dc


cache = dc.Cache('tmp')
cache["doc_pages"] = doc_pages
cache["text_encoder"] = text_encoder
cache["nn"] = nn

# %%
# Defining the Agent
# ~~~~~~~~~~~~~~~~~~
#
# We defined our pydantic-ai Agent with its retrieve function set as a tool.
# Notice how pydantic-ai enables you to specify a schema for the dependency ``Deps``,
# which is used as a ``RunContext`` during tool execution.
#
# For this example, we use OpenAI GPT-4o-mini rather than Llama3.3-70B with Groq's free
# tier as Groq currently struggles with tool calling.
import logfire
from dotenv import load_dotenv
from dataclasses import dataclass
from pydantic_ai import RunContext
from pydantic_ai.agent import Agent
import nest_asyncio


# Load the 'OPENAI_API_KEY' variable environment from a source file.
load_dotenv()

# Enable nested event loop in jupyter notebooks to run pydantic-ai
# asynchronous coroutines.
nest_asyncio.apply()

# Some boilerplate around logging.
logfire.configure(scrubbing=False)

@dataclass
class Deps:
    text_encoder: TextEncoder
    nn: NearestNeighbors
    doc_pages: pd.DataFrame

system_prompt = (
    "You are a documentation assistant. Your objective is to answer user questions "
    "by retrieving the right articles in the documentation. "
    "Don't look-up the documentation if the question is unrelated to LogFire "
    "or Pydantic. "
)

agent = Agent(
    'openai:gpt-4o-mini',
    system_prompt=system_prompt,
    deps_type=Deps,
)

def make_prompt(pages):
    return "\n\n".join(
        (
            "# " + pages["title"]
            + "\nDocumentation URL:" + pages["path"]
            + "\n\n" + pages["content"] + "\n"
        ).tolist()
    )

@agent.tool
async def retrieve(context: RunContext[Deps], search_query: str) -> str:
    """Retrieve documentation sections based on a search query.

    Args:
        context: The call context.
        search_query: The search query.
    """
    with logfire.span(f'create embedding for {search_query=}'):
        query_embedding = context.deps.text_encoder.transform(
            pd.Series([search_query]),
        )

    indices = (
        context.deps.nn.kneighbors(query_embedding, return_distance=False)
        .squeeze()
    )

    pages = context.deps.doc_pages.iloc[indices]
    doc_retrieved = make_prompt(pages)

    print(doc_retrieved)

    return doc_retrieved


# %%
# Finally, we define our main coroutine entry point to run the agent.

async def run_agent(question: str):
    """
    Entry point to run the agent and perform RAG based question answering.
    """
    logfire.info(f'Asking "{question}"')

    cache = dc.Cache('tmp')

    deps = Deps(
        text_encoder=cache["text_encoder"],
        nn=cache["nn"],
        doc_pages=cache["doc_pages"],
    )
    return await agent.run(question, deps=deps)


# %%
# Results
# ~~~~~~~
# We are now ready run our system! Logfire will generate logs for the different steps
# to help us observe the different internal steps.
import asyncio


answer = asyncio.run(
    run_agent("Can you summarize the roadmap for Logfire?")
)

# %%
# Let's now display the final response:
print(answer.data)

# %%
# The display below shows the sequence of messages from top to bottom (most recent).
#
# 1. The LLM correctly responded to our first query by calling a retrieval tool.
# 2. After retrieving the content queried by the LLM, we make another call with
#    this content.
# 3. Finally, the LLM sends back its text response, completing the message loop.

import json
from pprint import pprint


pprint(
    json.loads(answer.all_messages_json())
)

# %%
# Let's now observe the agent behavior for an unrelated query.
unrelated_answer = asyncio.run(
    run_agent("I'm a chef, explain how to bake a delicious brownie?")
)

# %%
# Since we specified in the agent system prompt not to perform a retrieval operation
# for unrelated questions, the agent responds with a plain text message indicating
# its inability to answer.

print(unrelated_answer.data)

# %%
# As expected, the message loop is smaller since the LLM didn't invoke the retrieve
# function, resulting in less latency and lower inference costs.
pprint(
    json.loads(unrelated_answer.all_messages_json())
)
# %%
# Finally, we cleanup the ``tmp`` diskcache folder.
from pathlib import Path


Path("tmp").rmdir()