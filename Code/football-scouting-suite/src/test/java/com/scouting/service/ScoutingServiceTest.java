package com.scouting.service;

import com.scouting.data.model.Player;
import com.scouting.data.repository.PlayerRepository;
import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.InjectMocks;
import org.mockito.Mock;
import org.mockito.junit.jupiter.MockitoExtension;

import java.util.Arrays;
import java.util.List;

import static org.mockito.Mockito.when;

@ExtendWith(MockitoExtension.class) // Abilita Mockito
class ScoutingServiceTest {

    // STUB creato automaticamente da Mockito (sostituisce la classe manuale)
    @Mock
    private PlayerRepository playerRepository;

    // DRIVER: Il service con lo stub iniettato dentro
    @InjectMocks
    private ScoutingService scoutingService;

    @Test
    void testGetAllPlayers() {
        Player p1 = new Player();
        p1.setName("Player A");
        Player p2 = new Player();
        p2.setName("Player B");
        
        when(playerRepository.findAll()).thenReturn(Arrays.asList(p1, p2));

        List<Player> result = scoutingService.getAllPlayers();

        Assertions.assertEquals(2, result.size());
        Assertions.assertEquals("Player A", result.get(0).getName());
    }
}