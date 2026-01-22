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

/**
 * Usiamo Mockito per simulare il Controller e verifichiamo che, quando la vista viene creata,
 * essa chiami il controller per ottenere i dati e popoli la griglia.
 * Utilizziamo la "Reflection" per sbirciare dentro i campi privati della vista (la Grid) e assicurarci che contenga
 * gli elementi attesi.
 */

@ExtendWith(MockitoExtension.class)
class MainViewTest {

    @Mock
    private ScoutingController scoutingController;
    
    @Test
    void testMainViewInitializationLoadsData() {
        List<Player> mockPlayers = new ArrayList<>();
        Player p = new Player();
        p.setName("Totti");
        mockPlayers.add(p);

        when(scoutingController.getAllPlayers()).thenReturn(mockPlayers);

        MainView view = new MainView(scoutingController);

        verify(scoutingController, times(1)).getAllPlayers();

        @SuppressWarnings("unchecked")
        Grid<Player> grid = (Grid<Player>) ReflectionTestUtils.getField(view, "grid");
        
        Assertions.assertNotNull(grid);
        
        @SuppressWarnings("unchecked")
        ListDataProvider<Player> dataProvider = (ListDataProvider<Player>) grid.getDataProvider();
        Assertions.assertEquals(1, dataProvider.getItems().size());
        Assertions.assertEquals("Totti", dataProvider.getItems().iterator().next().getName());
    }
}