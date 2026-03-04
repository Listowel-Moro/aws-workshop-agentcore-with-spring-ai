package com.example.agent;

import org.springframework.ai.chat.client.ChatClient;
import org.springframework.stereotype.Service;
import reactor.core.publisher.Flux;
import org.springaicommunity.agentcore.annotation.AgentCoreInvocation;

record ChatRequest(String prompt) {}

@Service
public class ChatService {
    private final ChatClient chatClient;
    private static final String SYSTEM_PROMPT = """
        You are a helpful AI agent for travel and expense management.
        Be friendly, helpful, and concise in your responses.
        """;

    public ChatService(ChatClient.Builder chatClientBuilder) {
        this.chatClient = chatClientBuilder
            .defaultSystem(SYSTEM_PROMPT)
            .build();
    }

    @AgentCoreInvocation
    public Flux<String> chat(ChatRequest request) {
        return chatClient.prompt().user(request.prompt()).stream().content();
    }
}
