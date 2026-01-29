#pragma semicolon 1
#include <sourcemod>
#include <l4d2_bile>

public Plugin myinfo = {
    name = "Bad Plugin",
    version = "1.0"
};

public void OnPluginStart()
{
    HookEvent("fake_event", Event_Fake);
}
