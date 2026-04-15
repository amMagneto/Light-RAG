from db_utils import create_application_logs, insert_application_logs, get_chat_history
from langchain_core.messages import HumanMessage, AIMessage
import uuid

def test_database_workflow():
    # 1. Setup: Create a fresh table and a unique session ID
    print("--- Starting DB Test ---")
    create_application_logs()
    test_session = f"test_{uuid.uuid4()}"
    
    # 2. Action: Insert a mock conversation turn
    user_q = "What is a first principle?"
    ai_a = "It is a basic assumption that cannot be deduced any further."
    insert_application_logs(test_session, user_q, ai_a, "test-model")
    print(f"Inserted test turn for session: {test_session}")

    # 3. Validation: Retrieve the history
    history = get_chat_history(test_session)
    
    # 4. Assertions: Prove the code is correct
    assert len(history) == 2, f"Expected 2 messages, got {len(history)}"
    
    # This is the critical test for your recent fix:
    assert isinstance(history[0], HumanMessage), "First message must be a HumanMessage object"
    assert isinstance(history[1], AIMessage), "Second message must be an AIMessage object"
    
    assert history[0].content == user_q
    assert history[1].content == ai_a

    print("--- Test Passed Successfully! ---")
    print(f"Retrieved {type(history[0]).__name__}: {history[0].content}")
    print(f"Retrieved {type(history[1]).__name__}: {history[1].content}")

if __name__ == "__main__":
    test_database_workflow()