import logging
from datetime import datetime

from dotenv import load_dotenv
from livekit import rtc
from livekit.agents import (
    Agent,
    AgentServer,
    AgentSession,
    JobContext,
    JobProcess,
    RunContext,
    cli,
    function_tool,
    inference,
    room_io,
)
from livekit.plugins import noise_cancellation, silero
from livekit.plugins.turn_detector.multilingual import MultilingualModel

from db import init_db, insert_appointment, select_appointments, select_appointments_by_patient

logger = logging.getLogger("agent")

load_dotenv(".env.local")


class Assistant(Agent):
    def __init__(self) -> None:
        super().__init__(
            instructions=f"""You are a helpful voice AI assistant that schedules appointments for a medical practice called Robot Medical Group.
            The user is interacting with you via voice, even if you perceive the conversation as text.
            You eagerly assist users with their questions by scheduling appointments or providing information from your extensive knowledge.
            Your responses are concise, to the point, and without any complex formatting or punctuation including emojis, asterisks, or other symbols.
            You are curious, friendly, and have a sense of humor.""",
        )
    
    async def on_enter(self) -> None:
        await self.session.generate_reply(
            instructions="Greet the user, thank them for calling, and ask how you can help.",
            allow_interruptions=True,
        )
    
    @function_tool
    async def get_doctors(self, context: RunContext) -> list[dict]:
        """
        Use this tool to get the list of doctors available for appointments.
        """
        return [
            "Dr. Smith",
            "Dr. Williams",
            "Dr. Brown",
        ]

    @function_tool
    async def get_working_hours(self, context: RunContext) -> list[dict]:
        """
        Use this tool to get the office hours of the medical practice.
        """
        return [
            "Monday - Friday: 9:00 AM - 5:00 PM",
            "Saturday - Sunday: 10:00 AM - 4:00 PM",
        ]

    @function_tool
    async def get_current_date_and_time(self, context: RunContext) -> list[dict]:
        """
        Use this tool to get the current date and time, in particular when a caller
        requests an appointment relative to the current date and time, 
        e.g. "tomorrow", "next week", "in an hour", etc.

        Returns:
            Date and time string in the format "YYYY-MM-DD HH:MM:SS"
        """
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    @function_tool
    async def add_appointment(self, context: RunContext, patient_name: str, doctor_name: str, scheduled_at: str, summary: str) -> None:
        """
        Use this tool to schedule a new appointment for a patient.

        Make sure to use the get_current_date_and_time tool if the caller requests an appointment relative 
        to the current date and time.

        Please also make sure the caller provides their name, preferred doctor, and the reason for the appointment.

        If the user doesn't have a preferred doctor, you can use the get_doctors tool to get the list of doctors available for appointments.
        
        Make sure to use the get_office_hours tool to ensure that the appointment is scheduled while the office is open. Only read back the hours
        to the caller if they ask about office hours or if they try to schedule an appointment outside of the office hours.

        When the appointment is scheduled, please read back the appointment details, including the full date and time.
        When reading the time back to the user, please provide it in AM/PM format, not 24-hour format.

        Args:
            patient_name: The name of the patient
            doctor_name: The name of the doctor
            scheduled_at: The date and time of the appointment in YYYY-MM-DD HH:MM:SS format
            summary: A brief summary of the appointment

        Returns:
            Appointment details if scheduled successfully, error message otherwise
        """
        
        appointment, error = insert_appointment(patient_name, doctor_name, scheduled_at, summary)
        if error:
            logger.error(f"Error inserting appointment: {error}")
            return f"Error inserting appointment: {error}"
        return f"Appointment added successfully. Created: {appointment}"

    @function_tool
    async def get_appointments_for_patient(self, context: RunContext, patient_name: str) -> list[dict]:
        """
        Use this tool to get all appointments for a patient.

        Args:
            patient_name: The name of the patient

        Returns:
            A list of appointments for the patient
        """

        return select_appointments_by_patient(patient_name)


server = AgentServer()


def prewarm(proc: JobProcess):
    init_db()
    proc.userdata["vad"] = silero.VAD.load()


server.setup_fnc = prewarm


@server.rtc_session(agent_name="appointment-scheduler-agent")
async def appointment_scheduler_agent(ctx: JobContext):
    # Logging setup
    # Add any other context you want in all log entries here
    ctx.log_context_fields = {
        "room": ctx.room.name,
    }

    # Set up a voice AI pipeline using OpenAI, Cartesia, Deepgram, and the LiveKit turn detector
    session = AgentSession(
        # Speech-to-text (STT) is your agent's ears, turning the user's speech into text that the LLM can understand
        # See all available models at https://docs.livekit.io/agents/models/stt/
        stt=inference.STT(model="deepgram/nova-3", language="multi"),
        # A Large Language Model (LLM) is your agent's brain, processing user input and generating a response
        # See all available models at https://docs.livekit.io/agents/models/llm/
        llm=inference.LLM(model="openai/gpt-4.1-mini"),
        # Text-to-speech (TTS) is your agent's voice, turning the LLM's text into speech that the user can hear
        # See all available models as well as voice selections at https://docs.livekit.io/agents/models/tts/
        tts=inference.TTS(
            model="cartesia/sonic-3", voice="9626c31c-bec5-4cca-baa8-f8ba9e84c8bc"
        ),
        # VAD and turn detection are used to determine when the user is speaking and when the agent should respond
        # See more at https://docs.livekit.io/agents/build/turns
        turn_detection=MultilingualModel(),
        vad=ctx.proc.userdata["vad"],
        # allow the LLM to generate a response while waiting for the end of turn
        # See more at https://docs.livekit.io/agents/build/audio/#preemptive-generation
        preemptive_generation=True,
    )

    # To use a realtime model instead of a voice pipeline, use the following session setup instead.
    # (Note: This is for the OpenAI Realtime API. For other providers, see https://docs.livekit.io/agents/models/realtime/))
    # 1. Install livekit-agents[openai]
    # 2. Set OPENAI_API_KEY in .env.local
    # 3. Add `from livekit.plugins import openai` to the top of this file
    # 4. Use the following session setup instead of the version above
    # session = AgentSession(
    #     llm=openai.realtime.RealtimeModel(voice="marin")
    # )

    # # Add a virtual avatar to the session, if desired
    # # For other providers, see https://docs.livekit.io/agents/models/avatar/
    # avatar = hedra.AvatarSession(
    #   avatar_id="...",  # See https://docs.livekit.io/agents/models/avatar/plugins/hedra
    # )
    # # Start the avatar and wait for it to join
    # await avatar.start(session, room=ctx.room)

    # Start the session, which initializes the voice pipeline and warms up the models
    await session.start(
        agent=Assistant(),
        room=ctx.room,
        room_options=room_io.RoomOptions(
            audio_input=room_io.AudioInputOptions(
                noise_cancellation=lambda params: (
                    noise_cancellation.BVCTelephony()
                    if params.participant.kind
                    == rtc.ParticipantKind.PARTICIPANT_KIND_SIP
                    else noise_cancellation.BVC()
                ),
            ),
        ),
    )

    # Join the room and connect to the user
    await ctx.connect()


if __name__ == "__main__":
    cli.run_app(server)
