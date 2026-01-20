package com.scouting.data.model;

import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;

class PlayerTest {

    // DRIVER: Il test simula il client che usa l'oggetto Player
    @Test
    void testPlayerAttributes() {
        // Setup
        Player player = new Player();
        Long expectedId = 1L;
        String expectedName = "Messi";
        Double expectedGoals = 0.95;

        player.setId(expectedId);
        player.setName(expectedName);
        player.setGoalsP90(expectedGoals);

        Assertions.assertEquals(expectedId, player.getId());
        Assertions.assertEquals(expectedName, player.getName());
        Assertions.assertEquals(expectedGoals, player.getGoalsP90());
    }
}