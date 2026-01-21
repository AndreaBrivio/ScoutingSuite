package com.scouting.ui;

import com.scouting.data.model.Player;
import com.scouting.service.PlayerFilterRequest;
import com.scouting.service.ScoutingService;
import com.scouting.service.StatFilterCriteria;

import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import org.springframework.stereotype.Component;

import java.util.List;

/**
 * Controller (MVC): Gestisce l'interazione tra la View (UI) e il Model (Service/Data).
 * Agisce anche come Facade nascondendo la complessità del DTO alla View.
 */
@Component
public class ScoutingController {

    // Utilizziamo Log4j come richiesto
    private static final Logger logger = LogManager.getLogger(ScoutingController.class);
    
    private final ScoutingService scoutingService;

    public ScoutingController(ScoutingService scoutingService) {
        this.scoutingService = scoutingService;
    }

    /**
     * Recupera tutti i giocatori iniziali.
     */
    public List<Player> getAllPlayers() {
        return scoutingService.getAllPlayers();
    }

    /**
     * Riceve i parametri grezzi dalla View, crea il DTO (PlayerFilterRequest)
     * e invoca il servizio di business logic.
     */
    public List<Player> searchPlayers(
            Integer minAge, Integer maxAge,
            String name, String squad,
            String competition, String nation,
            String position,
            List<StatFilterCriteria> dynamicFilters) {

        // Log dell'azione utente (Controller responsibility)
        logger.info("Ricerca avviata dall'utente - Filtri: Name='{}', Squad='{}', DynamicFilters={}", 
                    name, squad, (dynamicFilters != null ? dynamicFilters.size() : 0));

        // Creazione del DTO "PlayerFilterRequest" (che hai creato al Passo 1)
        PlayerFilterRequest request = new PlayerFilterRequest(
                minAge, maxAge, name, squad, competition, nation, position, dynamicFilters
        );

        // Delega al Model
        return scoutingService.findPlayersByCriteria(request);
    }
}