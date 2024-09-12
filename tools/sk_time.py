from semantic_kernel.functions.kernel_function_decorator import kernel_function
from tools.bingsearch import bing_search
import datetime

class DateTimePlugin:
    @kernel_function(name="date_time_tool", description="use this tool to get the current date and time")
    def date_time_tool(self, query:str) -> datetime:
        return datetime.datetime.now()