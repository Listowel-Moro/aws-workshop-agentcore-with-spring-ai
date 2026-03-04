package com.example.agent;

import java.util.ArrayList;
import java.util.List;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springaicommunity.agentcore.annotation.AgentCoreInvocation;
import org.springaicommunity.agentcore.context.AgentCoreContext;
import org.springaicommunity.agentcore.context.AgentCoreHeaders;
import org.springaicommunity.agentcore.memory.longterm.AgentCoreMemory;
import org.springframework.ai.chat.client.ChatClient;
import org.springframework.ai.chat.client.advisor.api.Advisor;
import org.springframework.ai.chat.memory.ChatMemory;
import org.springframework.stereotype.Service;
import reactor.core.publisher.Flux;
import org.springframework.ai.chat.client.advisor.vectorstore.QuestionAnswerAdvisor;
import org.springframework.ai.vectorstore.VectorStore;
import org.springaicommunity.agentcore.artifacts.ArtifactStore;
import org.springaicommunity.agentcore.artifacts.GeneratedFile;
import org.springaicommunity.agentcore.artifacts.SessionConstants;
import org.springaicommunity.agentcore.browser.BrowserArtifacts;
import org.springframework.ai.tool.ToolCallbackProvider;
import org.springframework.beans.factory.annotation.Qualifier;

record ChatRequest(String prompt) {}

@Service
public class ChatService {

    private static final Logger logger = LoggerFactory.getLogger(ChatService.class);

    private final ChatClient chatClient;
    private final ArtifactStore<GeneratedFile> browserArtifactStore;

    private static final String SYSTEM_PROMPT = """
        You are a helpful AI agent for travel and expense management.
        Be friendly, helpful, and concise in your responses.
        """;

    public ChatService(AgentCoreMemory agentCoreMemory,
        VectorStore kbVectorStore,
        WebGroundingTools webGroundingTools,
        ContextAdvisor contextAdvisor,
        @Qualifier("browserToolCallbackProvider") ToolCallbackProvider browserTools,
        @Qualifier("browserArtifactStore") ArtifactStore<GeneratedFile> browserArtifactStore,
        ChatClient.Builder chatClientBuilder) {

        List<Advisor> advisors = new ArrayList<>();

        // Memory (STM + LTM)
        advisors.addAll(agentCoreMemory.advisors);
        logger.info("Memory enabled: {} advisors", agentCoreMemory.advisors.size());

         // Knowledge Base (RAG)
        if (kbVectorStore != null) {
            advisors.add(QuestionAnswerAdvisor.builder(kbVectorStore).build());
            logger.info("KB RAG enabled");
        }

        // ContextAdvisor
        advisors.add(contextAdvisor);
		logger.info("Context Advisor enabled");

        // Tools
        List<Object> localTools = new ArrayList<>();
        if (webGroundingTools != null) {
            localTools.add(webGroundingTools);
			logger.info("Web Grounding enabled");
        }

                // Browser
        this.browserArtifactStore = browserArtifactStore;

        // Tool Callback Providers
        List<ToolCallbackProvider> toolCallbackProviders = new ArrayList<>();
        if (browserTools != null) {
            toolCallbackProviders.add(browserTools);
            logger.info("Browser enabled");
        }

        this.chatClient = chatClientBuilder
            .defaultSystem(SYSTEM_PROMPT)
            .defaultAdvisors(advisors.toArray(new Advisor[0]))
            .defaultTools(localTools.toArray())
            .defaultToolCallbacks(toolCallbackProviders.toArray(new ToolCallbackProvider[0]))
            .build();

            
    }

    @AgentCoreInvocation
    public Flux<String> chat(ChatRequest request, AgentCoreContext context) {
        return chat(request.prompt(), getConversationId(context));
    }

    private Flux<String> chat(String prompt, String sessionId) {
        return chatClient.prompt().user(prompt)
            .advisors(a -> a.param(ChatMemory.CONVERSATION_ID, sessionId))
            .stream().content()
            .concatWith(Flux.defer(() -> appendScreenshots(sessionId)))
            .contextWrite(ctx -> ctx.put(SessionConstants.SESSION_ID_KEY, sessionId));
    }

    private String getConversationId(AgentCoreContext context) {
        return context.getHeader(AgentCoreHeaders.SESSION_ID);
    }

        private Flux<String> appendScreenshots(String sessionId) {
        if (browserArtifactStore == null) {
            return Flux.empty();
        }
        List<GeneratedFile> screenshots = browserArtifactStore.retrieve(sessionId);
        if (screenshots == null || screenshots.isEmpty()) {
            return Flux.empty();
        }
        return Flux.just(formatScreenshotsAsMarkdown(screenshots));
    }

    private String formatScreenshotsAsMarkdown(List<GeneratedFile> screenshots) {
        StringBuilder sb = new StringBuilder();
        for (GeneratedFile screenshot : screenshots) {
            sb.append("\n\n![Screenshot of ")
                .append(BrowserArtifacts.url(screenshot).orElse("unknown"))
                .append("](")
                .append(screenshot.toDataUrl())
                .append(")");
        }
        return sb.toString();
    }
}