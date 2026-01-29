#pragma semicolon 1
#pragma newdecls required

#include <sourcemod>
#include <sdktools>
#include <sdkhooks>
#include <l4d2_infected_random>

public Plugin myinfo =
{
    name = "Random SI on Last Generator",
    author = "Developer",
    description = "Spawns a random special infected when the last generator is started",
    version = "1.0",
    url = ""
};

public void OnPluginStart()
{
    HookEvent("generator_final_start", Event_GeneratorFinalStart);
}

public void Event_GeneratorFinalStart(Event event, const char[] name, bool dontBroadcast)
{
    // Check if we are in a finale (this event is only fired in finales)
    if (IsFinaleInProgress())
    {
        // Spawn a random special infected
        SpawnRandomSpecialInfected();
    }
}

bool IsFinaleInProgress()
{
    // Check if the map is a finale (finale maps have "final" in their name)
    char mapName[64];
    GetMapName(mapName, sizeof(mapName));

    return (StrContains(mapName, "final") != -1);
}

void SpawnRandomSpecialInfected()
{
    // Use the L4D2 Infected Random library to spawn a random SI
    int siType = GetRandomSpecialInfected();
    if (siType == -1)
    {
        LogError("Failed to get a random special infected type.");
        return;
    }

    // Spawn the special infected
    int infected = CreateEntityByName("infected");
    if (infected == -1)
    {
        LogError("Failed to create infected entity.");
        return;
    }

    // Set the infected type
    DispatchKeyValue(infected, "m_iClass", siType);
    DispatchSpawn(infected);

    // Set the infected to be a special infected
    SetEntProp(infected, Prop_Send, "m_bIsSpecialInfected", 1);

    // Teleport the infected to a random location
    float origin[3];
    GetRandomSpawnLocation(origin);
    TeleportEntity(infected, origin, NULL_VECTOR, NULL_VECTOR);

    // Emit a sound
    EmitSoundToAll("ambient/levels/labs/electric_explosion1.wav");

    // Log the spawn
    char siName[64];
    GetSpecialInfectedName(siType, siName, sizeof(siName));
    PrintToChatAll("\x04[Random SI] \x01A %s has spawned!", siName);
}

void GetRandomSpawnLocation(float origin[3])
{
    // Get a random location within the map bounds
    float min[3], max[3];
    GetWorldBounds(min, max);

    origin[0] = RandomFloat(min[0], max[0]);
    origin[1] = RandomFloat(min[1], max[1]);
    origin[2] = 0.0; // Set Z to ground level
}

void GetSpecialInfectedName(int siType, char[] buffer, int maxlen)
{
    switch (siType)
    {
        case 0: StrCopy(buffer, maxlen, "Smoker");
        case 1: StrCopy(buffer, maxlen, "Boomer");
        case 2: StrCopy(buffer, maxlen, "Hunter");
        case 3: StrCopy(buffer, maxlen, "Spitter");
        case 4: StrCopy(buffer, maxlen, "Jockey");
        case 5: StrCopy(buffer, maxlen, "Charger");
        case 6: StrCopy(buffer, maxlen, "Screamer");
        case 7: StrCopy(buffer, maxlen, "Spitter");
        case 8: StrCopy(buffer, maxlen, "Screamer");
        case 9: StrCopy(buffer, maxlen, "Screamer");
        case 10: StrCopy(buffer, maxlen, "Screamer");
        case 11: StrCopy(buffer, maxlen, "Screamer");
        case 12: StrCopy(buffer, maxlen, "Screamer");
        case 13: StrCopy(buffer, maxlen, "Screamer");
        case 14: StrCopy(buffer, maxlen, "Screamer");
        case 15: StrCopy(buffer, maxlen, "Screamer");
        case 16: StrCopy(buffer, maxlen, "Screamer");
        case 17: StrCopy(buffer, maxlen, "Screamer");
        case 18: StrCopy(buffer, maxlen, "Screamer");
        case 19: StrCopy(buffer, maxlen, "Screamer");
        case 20: StrCopy(buffer, maxlen, "Screamer");
        case 21: StrCopy(buffer, maxlen, "Screamer");
        case 22: StrCopy(buffer, maxlen, "Screamer");
        case 23: StrCopy(buffer, maxlen, "Screamer");
        case 24: StrCopy(buffer, maxlen, "Screamer");
        case 25: StrCopy(buffer, maxlen, "Screamer");
        case 26: StrCopy(buffer, maxlen, "Screamer");
        case 27: StrCopy(buffer, maxlen, "Screamer");
        case 28: StrCopy(buffer, maxlen, "Screamer");
        case 29: StrCopy(buffer, maxlen, "Screamer");
        case 30: StrCopy(buffer, maxlen, "Screamer");
        case 31: StrCopy(buffer, maxlen, "Screamer");
        case 32: StrCopy(buffer, maxlen, "Screamer");
        case 33: StrCopy(buffer, maxlen, "Screamer");
        case 34: StrCopy(buffer, maxlen, "Screamer");
        case 35: StrCopy(buffer, maxlen, "Screamer");
        case 36: StrCopy(buffer, maxlen, "Screamer");
        case 37: StrCopy(buffer, maxlen, "Screamer");
        case 38: StrCopy(buffer, maxlen, "Screamer");
        case 39: StrCopy(buffer, maxlen, "Screamer");
        case 40: StrCopy(buffer, maxlen, "Screamer");
        case 41: StrCopy(buffer, maxlen, "Screamer");
        case 42: StrCopy(buffer, maxlen, "Screamer");
        case 43: StrCopy(buffer, maxlen, "Screamer");
        case 44: StrCopy(buffer, maxlen, "Screamer");
        case 45: StrCopy(buffer, maxlen, "Screamer");
        case 46: StrCopy(buffer, maxlen, "Screamer");
        case 47: StrCopy(buffer, maxlen, "Screamer");
        case 48: StrCopy(buffer, maxlen, "Screamer");
        case 49: StrCopy(buffer, maxlen, "Screamer");
        case 50: StrCopy(buffer, maxlen, "Screamer");
        case 51: StrCopy(buffer, maxlen, "Screamer");
        case 52: StrCopy(buffer, maxlen, "Screamer");
        case 53: StrCopy(buffer, maxlen, "Screamer");
        case 54: StrCopy(buffer, maxlen, "Screamer");
        case 55: StrCopy(buffer, maxlen, "Screamer");
        case 56: StrCopy(buffer, maxlen, "Screamer");
        case 57: StrCopy(buffer, maxlen, "Screamer");
        case 58: StrCopy(buffer, maxlen, "Screamer");
        case 59: StrCopy(buffer, maxlen, "Screamer");
        case 60: StrCopy(buffer, maxlen, "Screamer");
        case 61: StrCopy(buffer, maxlen, "Screamer");
        case 62: StrCopy(buffer, maxlen, "Screamer");
        case 63: StrCopy(buffer, maxlen, "Screamer");
        case 64: StrCopy(buffer, maxlen, "Screamer");
        case 65: StrCopy(buffer, maxlen, "Screamer");
        case 66: StrCopy(buffer, maxlen, "Screamer");
        case 67: StrCopy(buffer, maxlen, "Screamer");
        case 68: StrCopy(buffer, maxlen, "Screamer");
        case 69: StrCopy(buffer, maxlen, "Screamer");
        case 70: StrCopy(buffer, maxlen, "Screamer");
        case 71: StrCopy(buffer, maxlen, "Screamer");
        case 72: StrCopy(buffer, maxlen, "Screamer");
        case 73: StrCopy(buffer, maxlen, "Screamer");
        case 74: StrCopy(buffer, maxlen, "Screamer");
        case 75: StrCopy(buffer, maxlen, "Screamer");
        case 76: StrCopy(buffer, maxlen, "Screamer");
        case 77: StrCopy(buffer, maxlen, "Screamer");
        case 78: StrCopy(buffer, maxlen, "Screamer");
        case 79: StrCopy(buffer, maxlen, "Screamer");
        case 80: StrCopy(buffer, maxlen, "Screamer");
        case 81: StrCopy(buffer, maxlen, "Screamer");
        case 82: StrCopy(buffer, maxlen, "S