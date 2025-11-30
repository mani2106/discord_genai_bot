import discord
from discord.ext import commands
from discord import app_commands
from dotenv import load_dotenv
import os

load_dotenv()

# --- Setup ---
intents = discord.Intents.default()
intents.message_content = True
bot = commands.Bot(command_prefix="!", intents=intents)

file_loc = "./filestore"

def setup_filestore():
    os.makedirs(os.path.join(file_loc, "images"), exist_ok=True)
    os.makedirs(os.path.join(file_loc, "docs"), exist_ok=True)
    os.makedirs(os.path.join(file_loc, "embeds"), exist_ok=True)


# --- Example RAG pipeline (stub) ---
def rag_query(query: str) -> str:
    # TODO: Embed query, retrieve docs, run LLM
    return f"RAG answer for: {query}"

# --- Example Vision pipeline (stub) ---
def describe_image(file_path: str) -> str:
    # TODO: Load BLIP/CLIP model, generate caption
    return f"Description of uploaded image at {file_path}"

# --- Commands ---
@bot.tree.command(name="ask", description="Ask a question (RAG)")
async def ask(interaction: discord.Interaction, query: str):
    answer = rag_query(query)
    await interaction.response.send_message(answer)

@bot.tree.command(name="image", description="Upload an image for description")
async def image(interaction: discord.Interaction, attachment: discord.Attachment):
    # Save image locally
    file_path = os.path.join(file_loc, "images", attachment.filename)
    await attachment.save(file_path)

    caption = describe_image(file_path)
    await interaction.response.send_message(caption)

@bot.tree.command(name="help", description="Show usage instructions")
async def help_cmd(interaction: discord.Interaction):
    help_text = (
        "**Bot Commands:**\n"
        "• `/ask <query>` — Retrieve answers from knowledge base (RAG)\n"
        "• `/image <upload>` — Describe or tag an uploaded image\n"
        "• `/help` — Show this message\n"
        "• `/show_files` — Show uploaded files\n"
        "• `/ping` — Test command will always reply 'pong'\n"
    )
    await interaction.response.send_message(help_text)

@bot.tree.command(name="ping", description="Check bot responsiveness")
async def ping(interaction: discord.Interaction):
    await interaction.response.send_message("Pong!")

# command to show files uploaded by users
@bot.tree.command(name="show_files", description="Show uploaded files")
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
TOKEN = os.getenv("DISCORD_TOKEN")
GUILD_ID = os.getenv("GUILD_ID")  # Replace with your guild ID

@bot.event
async def on_ready():
    guild = discord.Object(id=int(GUILD_ID))
    await bot.tree.sync(guild=guild)
    setup_filestore()
    print(f"Logged in as {bot.user}")

bot.run(TOKEN)