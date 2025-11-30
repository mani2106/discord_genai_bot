import discord
from discord.ext import commands
from discord import app_commands
from dotenv import load_dotenv
import os, asyncio
from image_cap_flow import discord_img_llm as img_llm

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

# --- Run Bot ---

@bot.event
async def on_ready():
    guild = os.getenv("GUILD_ID")
    guild = discord.Object(id=int(guild))
    # await clear_global_commands(bot.tree)
    await bot.tree.sync(guild=guild)
    setup_filestore()
    print(f"Logged in as {bot.user}")

bot.run(TOKEN)