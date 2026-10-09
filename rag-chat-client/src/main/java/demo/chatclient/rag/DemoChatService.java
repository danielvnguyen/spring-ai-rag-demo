package demo.chatclient.rag;

import org.springframework.ai.chat.client.ChatClient;
import org.springframework.ai.document.Document;
import org.springframework.ai.vectorstore.SearchRequest;
import org.springframework.ai.vectorstore.VectorStore;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Service;

import java.util.List;
import java.util.stream.Collectors;

@Service
public class DemoChatService implements DemoChatClient {

    private static final String SYSTEM = """
            You answer the user's question using ONLY the context provided.
            The context is content fetched from web pages. If the context contains
            nothing relevant, say so plainly instead of guessing. Be concise.""";

    private final ChatClient chatClient;
    private final VectorStore vectorStore;
    private final int topK;
    private final String configuredModel;

    public DemoChatService(ChatClient.Builder chatClientBuilder,
                           VectorStore vectorStore,
                           @Value("${app.rag.top-k:5}") int topK,
                           @Value("${spring.ai.openai.chat.options.model}") String configuredModel) {
        this.chatClient = chatClientBuilder.defaultSystem(SYSTEM).build();
        this.vectorStore = vectorStore;
        this.topK = topK;
        this.configuredModel = configuredModel;
    }

    @SuppressWarnings("null")
    @Override
    public Answer askQuestion(Question question) {
        List<Document> hits = retrieve(question.question(), topK);
        String context = hits.stream()
                .map(Document::getText)
                .collect(Collectors.joining("\n\n"));

        var chatResponse = chatClient.prompt()
            .user(u -> u.text("""
                    Context:

                    {context}

                    Question: {question}""")
                    .param("context", context)
                    .param("question", question.question()))
            .call()
            .chatResponse();
        String text = chatResponse.getResult().getOutput().getText();
        String usedModel = chatResponse.getMetadata().getModel();
        if (usedModel == null || usedModel.isBlank()) {
            usedModel = configuredModel;
        }
        return new Answer(text == null ? "" : text, usedModel);
    }

    private List<Document> retrieve(String question, int k) {
        return vectorStore.similaritySearch(
                SearchRequest.builder().query(question).topK(k).build());
    }
}
