package com.scouting.UI;

import com.scouting.data.model.Player;
import com.scouting.service.ScoutingService;
import com.vaadin.flow.data.provider.ListDataProvider;
import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.Mock;
import org.mockito.junit.jupiter.MockitoExtension;
import org.springframework.test.util.ReflectionTestUtils;
import com.vaadin.flow.component.grid.Grid;

import java.util.ArrayList;
import java.util.List;

import static org.mockito.Mockito.*;

@ExtendWith(MockitoExtension.class)
class MainViewTest {

    @Mock
    private ScoutingService scoutingService;
    
    @Test
    void testMainViewInitializationLoadsData() {
        List<Player> mockPlayers = new ArrayList<>();
        Player p = new Player();
        p.setName("Totti");
        mockPlayers.add(p);

        // Quando la View chiama il service, restituiamo la lista mock
        when(scoutingService.getAllPlayers()).thenReturn(mockPlayers);

        // MainView chiama scoutingService.getAllPlayers() nel costruttore
        MainView view = new MainView(scoutingService);

        // Verifichiamo che il service sia stato chiamato 1 volta
        verify(scoutingService, times(1)).getAllPlayers();

        // Verifichiamo (tramite Reflection perché grid è privata) che la Grid abbia i dati
        @SuppressWarnings("unchecked")
        Grid<Player> grid = (Grid<Player>) ReflectionTestUtils.getField(view, "grid");
        
        Assertions.assertNotNull(grid);
        
        // Estraiamo i dati dalla Grid per vedere se c'è Totti
        // (Vaadin 24 usa DataProvider)
        ListDataProvider<Player> dataProvider = (ListDataProvider<Player>) grid.getDataProvider();
        Assertions.assertEquals(1, dataProvider.getItems().size());
        Assertions.assertEquals("Totti", dataProvider.getItems().iterator().next().getName());
    }
}