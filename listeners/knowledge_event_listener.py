from crewai.events import (
    KnowledgeQueryStartedEvent,
    KnowledgeQueryCompletedEvent,
    BaseEventListener
)
from crewai.events.event_bus import CrewAIEventsBus


class KnowledgeMonitoringListener(BaseEventListener):
    def setup_listeners(self, crewai_event_bus: CrewAIEventsBus) -> None:
        @crewai_event_bus.on(KnowledgeQueryStartedEvent)
        def on_knowledge_query_started(source, event: KnowledgeQueryStartedEvent) -> None:
            print(f"Source: {source}")
            print(f"knowledge query started: {event.agent_role}")

        @crewai_event_bus.on(KnowledgeQueryCompletedEvent)
        def on_knowledge_query_completed(source, event: KnowledgeQueryCompletedEvent) -> None:
            print(f"Source: {source}")
            print(f"knowledge query completed: {event.agent_role}")

listener = KnowledgeMonitoringListener()