from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI


def create_rag_chain(retriever):
    prompt = ChatPromptTemplate.from_template(
        """
        Answer the question using only the context below.
        If the answer is not present, say "I don't know".

        Context:
        {context}

        Question:
        {question}
        """
    )

    llm = ChatOpenAI(
        model="gpt-3.5-turbo",
        temperature=0
    )

    rag_chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
    )

    return rag_chain
