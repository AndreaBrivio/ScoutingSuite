package com.scouting.ui;

import com.scouting.data.model.Player;
import com.vaadin.flow.component.grid.Grid;
import com.vaadin.flow.data.provider.ListDataProvider;
import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.Mock;
import org.mockito.junit.jupiter.MockitoExtension;
import org.springframework.test.util.ReflectionTestUtils;

import java.util.ArrayList;
import java.util.List;

import static org.mockito.Mockito.*;

@ExtendWith(MockitoExtension.class)
class MainViewTest {

    // MODIFICA: Mockiamo il Controller, non più il Service direttamente
    @Mock
    private ScoutingController scoutingController;
    
    @Test
    void testMainViewInitializationLoadsData() {
        List<Player> mockPlayers = new ArrayList<>();
        Player p = new Player();
        p.setName("Totti");
        mockPlayers.add(p);

        // Quando la View chiama il controller, restituiamo la lista mock
        when(scoutingController.getAllPlayers()).thenReturn(mockPlayers);

        // MODIFICA: Iniettiamo il Controller nel costruttore
        MainView view = new MainView(scoutingController);

        // Verifichiamo che il Controller sia stato chiamato 1 volta
        verify(scoutingController, times(1)).getAllPlayers();

        // Verifichiamo (tramite Reflection perché grid è privata) che la Grid abbia i dati
        @SuppressWarnings("unchecked")
        Grid<Player> grid = (Grid<Player>) ReflectionTestUtils.getField(view, "grid");
        
        Assertions.assertNotNull(grid);
        
        // Estraiamo i dati dalla Grid per vedere se c'è Totti
        @SuppressWarnings("unchecked")
        ListDataProvider<Player> dataProvider = (ListDataProvider<Player>) grid.getDataProvider();
        Assertions.assertEquals(1, dataProvider.getItems().size());
        Assertions.assertEquals("Totti", dataProvider.getItems().iterator().next().getName());
    }
}