package com.scouting.data.repository;

import com.scouting.data.model.Player;
import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.autoconfigure.orm.jpa.DataJpaTest;
import java.util.Optional;

/**
 * Questo è un test di integrazione per il layer dei dati (Data Access Layer).
 * Utilizza l'annotazione @DataJpaTest che avvia un vero database H2 in memoria (non un mock).
 * L'obiettivo è verificare che la nostra entità Player sia mappata correttamente sulle tabelle del DB e che
 * le operazioni di salvataggio e recupero funzionino realmente. È fondamentale per garantire che l'SQL generato
 * da Hibernate sia corretto.
 */

@DataJpaTest
class PlayerRepositoryTest {

    @Autowired
    private PlayerRepository playerRepository;

    @Test
    void testSaveAndFindPlayer() {
        Player p = new Player();
        p.setName("Roberto Baggio");
        p.setNation("Italy");
        p.setGoals(10);
        
        Player savedPlayer = playerRepository.save(p);
        
        Assertions.assertNotNull(savedPlayer.getId(), "L'ID dovrebbe essere generato automaticamente dal DB");
        
        Optional<Player> retrieved = playerRepository.findById(savedPlayer.getId());
        Assertions.assertTrue(retrieved.isPresent());
        Assertions.assertEquals("Roberto Baggio", retrieved.get().getName());
        Assertions.assertEquals(10, retrieved.get().getGoals());
    }
}