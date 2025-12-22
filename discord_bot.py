import discord
from discord.ext import commands
from discord import app_commands
from dotenv import load_dotenv
import os, asyncio
from image_cap_flow import discord_img_llm as img_llm
from rag_system.rag_system import DiscordRAGSystem

load_dotenv()

TOKEN = os.getenv("DISCORD_TOKEN")
guild = os.getenv("GUILD_ID")
guild = discord.Object(id=int(guild))

# --- Setup ---
intents = discord.Intents.default()
intents.message_content = True
bot = commands.Bot(command_prefix="!", intents=intents)

file_loc = "./filestore"

def setup_filestore():
    os.makedirs(os.path.join(file_loc, "images"), exist_ok=True)
    os.makedirs(os.path.join(file_loc, "docs"), exist_ok=True)
    os.makedirs(os.path.join(file_loc, "embeds"), exist_ok=True)

# Initialize RAG system
rag_system = None

def init_rag_system():
    """Initialize the RAG system."""
    global rag_system
    try:
        # Check if required environment variables are set
        required_env_vars = ['OPENROUTER_API_KEY', 'OPENROUTER_API_BASE', 'OPENROUTER_MODEL', 'OPENROUTER_EMBED_MODEL']
        missing_vars = [var for var in required_env_vars if not os.getenv(var)]

        if missing_vars:
            print(f"Warning: Missing environment variables for RAG system: {', '.join(missing_vars)}")
            print("RAG commands will not be available until these are configured.")
            rag_system = None
            return

        rag_system = DiscordRAGSystem(storage_path=file_loc)
        print("RAG system initialized successfully")
    except Exception as e:
        print(f"Failed to initialize RAG system: {e}")
        print("RAG commands will not be available.")
        rag_system = None

def split_into_chunks(text: str, limit: int = 2000) -> list[str]:
    """Split text into chunks of at most `limit` characters."""
    return [text[i:i+limit] for i in range(0, len(text), limit)]

async def clear_global_commands(tree: discord.app_commands.CommandTree):
    # Remove all global commands from the local tree
    for cmd in list(tree.get_commands()):  # global commands only
        tree.remove_command(cmd.name)      # removes global variant

    # Sync globally to push the cleared state
    await tree.sync()  # global sync (propagates with delay)

@bot.tree.command(name="image", description="Upload an image for description", guild=guild)
async def image(interaction: discord.Interaction, attachment: discord.Attachment):
    await interaction.response.defer(thinking=False, ephemeral=False)

    # Save image locally
    try:
        file_path = os.path.join(file_loc, "images", attachment.filename)
        await attachment.save(file_path)

        # Start a conversation for this user with the uploaded image (runs in thread)
        session_id = str(interaction.user.id)
        # caption = await asyncio.to_thread(img_llm.start_conversation_with_image, session_id, file_path)
        caption = await asyncio.to_thread(img_llm.start_conversation_with_image, session_id, file_path)
    except Exception as e:
        await interaction.followup.send(f"Processing failed: {e}")
        return

    # Split messages if too long
    chunks = split_into_chunks(caption or "")

    # Prepare a small image preview by attaching the saved file and using an embed.
    # Send the first chunk together with the attached image so the user sees the image being described.
    if chunks:
        first_chunk = chunks.pop(0)
    else:
        first_chunk = "(No description generated.)"

    try:
        preview_file = discord.File(file_path, filename=attachment.filename)
        embed = discord.Embed(title="Image preview")
        embed.set_image(url=f"attachment://{attachment.filename}")
        await interaction.followup.send(first_chunk, file=preview_file, embed=embed)
    except Exception:
        # If attaching the file fails for any reason, still send the first chunk as text.
        await interaction.followup.send(first_chunk)

    # Send remaining chunks (if any) as plain followups
    for chunk in chunks:
        await interaction.followup.send(chunk)


@bot.tree.command(name="img_ask", description="Ask follow-up questions about the last uploaded image", guild=guild)
async def img_ask(interaction: discord.Interaction, query: str):
    # Defer the interaction so we can use followup.send() safely
    await interaction.response.defer(thinking=True)

    session_id = str(interaction.user.id)
    try:
        answer = await asyncio.to_thread(img_llm.ask, session_id, query)
    except Exception as e:
        await interaction.followup.send(f"Processing failed: {e}")
        return

    # Split messages if too long
    chunks = split_into_chunks(answer)
    for chunk in chunks:
        await interaction.followup.send(chunk)

@bot.tree.command(name="img_clear", description="Clear stored image/context for your session", guild=guild)
async def img_clear(interaction: discord.Interaction):
    session_id = str(interaction.user.id)
    await asyncio.to_thread(img_llm.clear, session_id)
    await interaction.response.send_message("Image/context cleared.")

@bot.tree.command(name="help", description="Show usage instructions", guild=guild)
async def help_cmd(interaction: discord.Interaction):
    help_text = (
        "**Bot Commands:**\n"
        "• `/image <upload>` — Describe or tag an uploaded image\n"
        "• `/help` — Show this message\n"
        "• `/show_files` — Show uploaded files\n"
        "• `/ping` — Test command will always reply 'pong'\n"
        "**Image Follow-up Commands:**\n"
        "• `/img_ask <query>` — Ask follow-up questions about the last uploaded image\n"
        "• `/img_clear` — Clear stored image/context for your session\n"
        "**Document RAG Commands:**\n"
        "• `/upload_doc <file>` — Upload a text document for AI querying\n"
        "• `/ask_docs <query>` — Ask questions about your uploaded documents\n"
        "• `/list_docs` — Show your uploaded documents\n"
        "• `/clear_docs` — Clear all your uploaded documents\n"
        "• `/rag_status` — Show RAG system status\n"
    )
    await interaction.response.send_message(help_text)

@bot.tree.command(name="ping", description="Check bot responsiveness", guild=guild)
async def ping(interaction: discord.Interaction):
    await interaction.response.send_message("Pong!")

# command to show files uploaded by users
@bot.tree.command(name="show_files", description="Show uploaded files", guild=guild)
async def show_files(interaction: discord.Interaction):
    files = []
    for root, dirs, filenames in os.walk(file_loc):
        for filename in filenames:
            files.append(os.path.join(root, filename))
    if files:
        file_list = "\n".join(files)
        await interaction.response.send_message(f"Uploaded files:\n{file_list}")
    else:
        await interaction.response.send_message("No files uploaded yet.")

# --- RAG System Commands ---

@bot.tree.command(name="upload_doc", description="Upload a text document for RAG processing", guild=guild)
async def upload_doc(interaction: discord.Interaction, attachment: discord.Attachment):
    """Upload and process a text document for RAG querying."""
    await interaction.response.defer(thinking=True, ephemeral=False)

    if rag_system is None:
        await interaction.followup.send("❌ RAG system is not available. Please contact an administrator.")
        return

    # Validate file type
    allowed_extensions = {'.txt', '.md', '.text'}
    file_extension = os.path.splitext(attachment.filename)[1].lower()

    if file_extension not in allowed_extensions:
        await interaction.followup.send(
            f"❌ **Unsupported file type: {file_extension}**\n\n"
            f"Supported formats: {', '.join(allowed_extensions)}\n"
            f"Please upload a text file in one of the supported formats."
        )
        return

    # Check file size (limit to 10MB)
    max_size = 10 * 1024 * 1024  # 10MB in bytes
    if attachment.size > max_size:
        await interaction.followup.send(
            f"❌ **File too large: {attachment.size / (1024*1024):.1f}MB**\n\n"
            f"Maximum file size: {max_size / (1024*1024):.0f}MB\n"
            f"Please upload a smaller file."
        )
        return

    try:
        # Create user-specific directory
        user_id = str(interaction.user.id)
        user_doc_dir = os.path.join(file_loc, "docs", user_id)
        os.makedirs(user_doc_dir, exist_ok=True)

        # Save file locally
        file_path = os.path.join(user_doc_dir, attachment.filename)
        await attachment.save(file_path)

        # Process with RAG system
        result = await rag_system.process_file_upload(file_path, attachment.filename, user_id)

        # Split response if too long
        chunks = split_into_chunks(result)
        for chunk in chunks:
            await interaction.followup.send(chunk)

    except Exception as e:
        await interaction.followup.send(f"❌ **Processing failed:** {str(e)}")

@bot.tree.command(name="ask_docs", description="Ask questions about uploaded documents", guild=guild)
async def ask_docs(interaction: discord.Interaction, query: str):
    """Query the uploaded documents and get AI-generated responses."""
    await interaction.response.defer(thinking=True, ephemeral=False)

    if rag_system is None:
        await interaction.followup.send("❌ RAG system is not available. Please contact an administrator.")
        return

    # Validate query length
    if len(query.strip()) < 3:
        await interaction.followup.send(
            "❌ **Query too short**\n\n"
            "Please provide a more detailed question (at least 3 characters)."
        )
        return

    if len(query) > 500:
        await interaction.followup.send(
            "❌ **Query too long**\n\n"
            "Please keep your question under 500 characters."
        )
        return

    try:
        user_id = str(interaction.user.id)

        # Process query
        result = await rag_system.query_documents(query, user_id)

        # Split response if too long
        chunks = split_into_chunks(result)
        for chunk in chunks:
            await interaction.followup.send(chunk)

    except Exception as e:
        await interaction.followup.send(f"❌ **Query failed:** {str(e)}")

@bot.tree.command(name="clear_docs", description="Clear all uploaded documents for your account", guild=guild)
async def clear_docs(interaction: discord.Interaction):
    """Clear all uploaded documents for the user."""
    if rag_system is None:
        await interaction.response.send_message("❌ RAG system is not available. Please contact an administrator.")
        return

    try:
        user_id = str(interaction.user.id)
        result = rag_system.clear_user_documents(user_id)
        await interaction.response.send_message(result)

    except Exception as e:
        await interaction.response.send_message(f"❌ **Clear failed:** {str(e)}")

@bot.tree.command(name="list_docs", description="Show your uploaded documents", guild=guild)
async def list_docs(interaction: discord.Interaction):
    """List all uploaded documents for the user."""
    if rag_system is None:
        await interaction.response.send_message("❌ RAG system is not available. Please contact an administrator.")
        return

    try:
        user_id = str(interaction.user.id)
        result = rag_system.list_user_documents(user_id)

        # Split response if too long
        chunks = split_into_chunks(result)
        await interaction.response.send_message(chunks[0])

        for chunk in chunks[1:]:
            await interaction.followup.send(chunk)

    except Exception as e:
        await interaction.response.send_message(f"❌ **List failed:** {str(e)}")

@bot.tree.command(name="rag_status", description="Show RAG system status (admin)", guild=guild)
async def rag_status(interaction: discord.Interaction):
    """Show RAG system status information."""
    if rag_system is None:
        await interaction.response.send_message(
            "❌ **RAG System Unavailable**\n\n"
            "The RAG system failed to initialize. This could be due to:\n"
            "• Missing OpenRouter API configuration\n"
            "• Invalid API credentials\n"
            "• Network connectivity issues\n\n"
            "Please contact an administrator."
        )
        return

    try:
        result = rag_system.get_system_status()
        await interaction.response.send_message(result)

    except Exception as e:
        await interaction.response.send_message(f"❌ **Status check failed:** {str(e)}")

# --- Run Bot ---

@bot.event
async def on_ready():
    guild = os.getenv("GUILD_ID")
    guild = discord.Object(id=int(guild))
    # await clear_global_commands(bot.tree)
    await bot.tree.sync(guild=guild)
    setup_filestore()
    init_rag_system()
    print(f"Logged in as {bot.user}")

bot.run(TOKEN)