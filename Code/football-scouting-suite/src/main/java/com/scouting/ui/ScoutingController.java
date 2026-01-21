package com.scouting.ui;

import com.scouting.data.model.Player;
import com.scouting.service.PlayerFilterRequest;
import com.scouting.service.ScoutingService;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import org.springframework.stereotype.Component;
import java.util.List;

/**
 * Questa classe agisce come il punto di ingresso per tutte le interazioni dell'utente, fungendo da intermediario tra
 * l'interfaccia grafica (Vaadin) e la logica di business (Service Layer). Nel pattern architetturale <strong>MVC (Model-View-Controller)</strong>,
 * è fondamentale separare chi "mostra" i dati da chi li "elabora".
 * <p>
 * Qui adottiamo anche un pattern <em>Facade</em>: la UI non deve sapere come costruire oggetti complessi o come
 * interrogare il database; deve solo chiedere "dammi i giocatori che corrispondono a questi criteri".
 * Utilizziamo la Dependency Injection di Spring per collegarci al Service, mantenendo il codice pulito e testabile.
 * <p>
 * Dalle analisi statiche (JDepend), questo componente risulta avere un'alta instabilità (100%), il che è perfettamente
 * normale e corretto per il layer più esterno: dipende da tutti gli altri moduli sottostanti, ma nessuno dipende da lui.
 */

@Component
public class ScoutingController {

    private static final Logger logger = LogManager.getLogger(ScoutingController.class);
    
    private final ScoutingService scoutingService;

    public ScoutingController(ScoutingService scoutingService) {
        this.scoutingService = scoutingService;
    }

    
    public List<Player> getAllPlayers() {
        return scoutingService.getAllPlayers();
    }


    public List<Player> searchPlayers(PlayerFilterRequest request) {

        logger.info("Ricerca avviata - Filtri: Name='{}', DynamicFilters={}", 
                    request.name(), 
                    request.statFilters() != null ? request.statFilters().size() : 0);

        return scoutingService.findPlayersByCriteria(request);
    }
}