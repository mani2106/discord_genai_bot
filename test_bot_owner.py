from discord.ext import commands
import discord

class Owner(commands.Cog, name="Owner"):
    def __init__(self, bot) -> None:
        self.bot = bot

    @commands.Cog.listener()
    async def on_message(self, message):
        if message.author.bot:
            return
        if (
        self.bot.user.mentioned_in(message)
        and message.content.strip() == f"<@{self.bot.user.id}> sync"
        ):
            await self.process_message(message)

    async def process_message(self, message):
        synced = await self.bot.tree.sync()
        await message.channel.send(f"Successfully synced {len(synced)} command(s).")

    async def setup(bot) -> None:
        await bot.add_cog(Owner(bot))
