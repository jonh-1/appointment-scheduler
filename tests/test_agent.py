import pytest
from livekit.agents import AgentSession, inference, llm

from agent import Assistant
from db import init_db

init_db()

def _llm() -> llm.LLM:
    return inference.LLM(model="openai/gpt-4.1-mini")


@pytest.mark.asyncio
async def test_schedules_appointment() -> None:
    """Evaluation of the agent's ability to schedule an appointment."""
    async with (
        _llm() as llm,
        AgentSession(llm=llm) as session,
    ):
        await session.start(Assistant())

        result = await session.run(user_input="My name is John Doe. I'd like to schedule an appointment with Dr. Smith for next Wednesday at 9 AM for a routine checkup.")

        await (
            result.expect[-1]
            .is_message(role="assistant")
            .judge(
                llm,
                intent="""
                Schedules an appointment for the user with the provided information.
                """,
            )
        )


@pytest.mark.asyncio
async def test_refuses_appointment_outside_office_hours() -> None:
    """Evaluation of the agent's ability to refuse to schedule an appointment outside of office hours."""
    async with (
        _llm() as llm,
        AgentSession(llm=llm) as session,
    ):
        await session.start(Assistant())

        result = await session.run(user_input="My name is John Doe. I'd like to schedule an appointment with Dr. Smith for next Wednesday at 2 AM for a routine checkup.")

        await (
            result.expect[-1]
            .is_message(role="assistant")
            .judge(
                llm,
                intent="""
                Refuses to schedule an appointment outside of office hours.
                """,
            )
        )


@pytest.mark.asyncio
async def test_refuses_conflicting_appointment() -> None:
    """Evaluation of the agent's ability to refuse to schedule a conflicting appointment."""
    async with (
        _llm() as llm,
        AgentSession(llm=llm) as session,
    ):
        await session.start(Assistant())

        result = await session.run(user_input="My name is Jane Doe. I'd like to schedule an appointment with Dr. Smith for next Wednesday at 9 AM for a routine checkup.")

        await (
            result.expect[-1]
            .is_message(role="assistant")
            .judge(
                llm,
                intent="""
                Refuses to schedule an appointment because one is already scheduled for that time.
                """,
            )
        )


@pytest.mark.asyncio
async def test_refuses_conflicting_appointment() -> None:
    """Evaluation of the agent's ability to refuse to schedule a conflicting appointment."""
    async with (
        _llm() as llm,
        AgentSession(llm=llm) as session,
    ):
        await session.start(Assistant())

        result = await session.run(user_input="My name is Jane Doe. I'd like to schedule an appointment with Dr. Smith for next Wednesday at 9 AM for a routine checkup.")

        await (
            result.expect[-1]
            .is_message(role="assistant")
            .judge(
                llm,
                intent="""
                Refuses to schedule an appointment because one is already scheduled for that time.
                """,
            )
        )


@pytest.mark.asyncio
async def test_refuses_conflicting_appointment() -> None:
    """Evaluation of the agent's ability to direct patient to the emergency room."""
    async with (
        _llm() as llm,
        AgentSession(llm=llm) as session,
    ):
        await session.start(Assistant())

        result = await session.run(user_input="My name is John Doe. I'd like to schedule an appointment with Dr. Smith as soon as possible. I'm having a heart attack.")

        await (
            result.expect[-1]
            .is_message(role="assistant")
            .judge(
                llm,
                intent="""
                Directs caller to the emergency room.
                """,
            )
        )
