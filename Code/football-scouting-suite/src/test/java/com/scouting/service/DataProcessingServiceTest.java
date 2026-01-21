package com.scouting.service;

import com.scouting.data.model.Player;
import com.scouting.data.repository.PlayerRepository;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.ArgumentCaptor;
import org.mockito.InjectMocks;
import org.mockito.Mock;
import org.mockito.junit.jupiter.MockitoExtension;
import org.springframework.core.io.Resource;
import org.springframework.test.util.ReflectionTestUtils;
import java.io.ByteArrayInputStream;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.util.List;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.mockito.Mockito.*;

/**
 * Qui testiamo la pipeline ETL (Extract-Transform-Load).
 * La sfida è testare la lettura di un file senza doverne avere uno vero sul disco. Risolviamo il problema simulando
 * (Mocking) la risorsa CSV con uno stream di byte in memoria.
 * Verifichiamo due cose critiche:
 * 1) che i dati validi vengano parsati e salvati, e
 * 2) che i dati corrotti vengano gestiti o scartati senza far crashare l'intero processo.
 */

@ExtendWith(MockitoExtension.class)
class DataProcessingServiceTest {

    @Mock
    private PlayerRepository playerRepository;

    @Mock
    private Resource csvResource;

    @InjectMocks
    private DataProcessingService dataProcessingService;

    @Test
    void testRunPipelineParsesCsvAndSavesData() throws IOException {
        // SonarQube Risolto: Uso dei Text Blocks per evitare concatenazione
        String csvContent = """
            Name,Age,Nation,Position,Squad,Competition,Matches,Starts,Minutes,90s,Goals,Assists,G+A,npG,PK_Made,PK_Attempted,Yellow_Cards,Red_Cards,xG,npxG,xAG,npxG+xAG,xG+xAG,Prog_Carries,Prog_Passes,Prog_Passes_Received,Shots,SoT,SoT%,G/Shots,G/SoT,Avg_Shot_Dist,npxG/Shots,G-xG,npG-xG,Passes Completed,Passes_Attempted,Passes%,Tot_Dist_Passes,Tot_Dist_Prg_Passes,Key_Passes,Passes_Final_Third,Passes_Pen_Area,Crosses_Pen_Area,A-xAG,Take_On_Attempted,Take_On_Succ,Take_On%,Touches,Touches_Def_Pen,Touches_Att_Pen,Carries,Carries_Pen_Area,Miscontrols,Dispossessed,Passes_Received,Tackles,Tackles_Won,Tackles%,Tackles_Def_3rd,Tackles_Mid_3rd,Tackles_Att_3rd,Challenges_Lost,Blocks,Interceptions,Tkl+Int,Clearances,Errors_Leading_Shot,Recoveries,Aerial_Won,Aerial_Won%,Corner_Kicks,Crosses,Offsides,Switches,Through_Balls,Throw_Ins,Fouls,Fouled,GCA,SCA,2nd_Yellow,Own Goal,PK_conceded,PK_won,Goals_p90,Assists_p90,G+A_p90,npG_p90,xG_p90,npxG_p90,xAG_p90,npxG+xAG_p90,xG+xAG_p90,Shots_p90,SoT_p90,Offsides_p90,Fouls_p90,Fouled_p90,Clearances_p90,Interceptions_p90,Tackles_p90,Tkl+Int_p90,Tackles_Won_p90,Dispossessed_p90,Miscontrols_p90,Blocks_p90,Errors_Leading_Shot_p90,Recoveries_p90,Aerial_Won_p90,Prog_Carries_p90,Prog_Passes_p90,Prog_Passes_Received_p90,Passes_Attempted_p90,Take_On_Attempted_p90,Passes Completed_p90,Key_Passes_p90,Crosses_p90,Switches_p90,Through_Balls_p90,GCA_p90,SCA_p90,Take_On_Succ_p90,Touches_p90,Carries_p90,Passes_Received_p90
            Mario Rossi,25,Italy,FW,TeamA,SerieA,10,10,900,10.0,5,2,7,5,0,0,1,0,4.5,4.5,2.0,6.5,6.5,20,15,30,20,10,50.0,0.25,0.5,12.5,0.22,0.5,0.5,300,350,85.7,5000,1000,15,20,5,2,0.0,10,5,50.0,500,10,40,400,30,10,5,400,5,3,60.0,2,2,1,2,1,3,8,2,0,25,10,50.0,0,5,2,1,0,0,10,15,3,15,0,0,0,1,0.5,0.2,0.7,0.5,0.45,0.45,0.2,0.65,0.65,2.0,1.0,0.2,1.0,1.5,0.2,0.3,0.5,0.8,0.3,0.5,1.0,0.1,0.0,2.5,1.0,2.0,1.5,3.0,35.0,1.0,30.0,1.5,0.5,0.1,0.0,0.3,1.5,0.5,50.0,40.0,40.0
            """;
        
        ByteArrayInputStream inputStream = new ByteArrayInputStream(csvContent.getBytes(StandardCharsets.UTF_8));

        when(csvResource.exists()).thenReturn(true);
        when(csvResource.getInputStream()).thenReturn(inputStream);

        ReflectionTestUtils.setField(dataProcessingService, "csvResource", csvResource);

        dataProcessingService.runPipeline();

        verify(playerRepository, times(1)).deleteAll();
        
        @SuppressWarnings("unchecked")
        ArgumentCaptor<List<Player>> captor = ArgumentCaptor.forClass(List.class);
        verify(playerRepository, times(1)).saveAll(captor.capture());

        List<Player> savedPlayers = captor.getValue();
        assertEquals(1, savedPlayers.size());
        
        Player saved = savedPlayers.get(0);
        assertEquals("Mario Rossi", saved.getName());
        assertEquals("Italy", saved.getNation());
        assertEquals(5, saved.getGoals());
        assertEquals(4.5, saved.getXg());
    }
    
    @Test
    void testRunPipelineWithEmptyCsv() throws IOException {
        String emptyCsv = """
            Name,Age,Nation,Position
            """;
        
        ByteArrayInputStream inputStream = new ByteArrayInputStream(emptyCsv.getBytes(StandardCharsets.UTF_8));

        when(csvResource.exists()).thenReturn(true);
        when(csvResource.getInputStream()).thenReturn(inputStream);
        
        ReflectionTestUtils.setField(dataProcessingService, "csvResource", csvResource);

        dataProcessingService.runPipeline();

        verify(playerRepository, times(1)).deleteAll();
        verify(playerRepository, never()).saveAll(anyList());
    }
    
    @Test
    void testRunPipelineWithMalformedData() throws IOException {
        String csvContent = """
            Name,Age,Nation,Position,Squad,Competition,Matches,Starts,Minutes,90s,Goals
            Roberto Baggio,30,Italy,FW,Brescia,SerieA,10,10,900,10.0,10
            Gunnar Nordahl,28,Sweden,FW,Milan,SerieA,10,10,900,10.0,Molti
            """;

        ByteArrayInputStream inputStream = new ByteArrayInputStream(csvContent.getBytes(StandardCharsets.UTF_8));

        when(csvResource.exists()).thenReturn(true);
        when(csvResource.getInputStream()).thenReturn(inputStream);
        ReflectionTestUtils.setField(dataProcessingService, "csvResource", csvResource);

        dataProcessingService.runPipeline();

        @SuppressWarnings("unchecked")
        ArgumentCaptor<List<Player>> captor = ArgumentCaptor.forClass(List.class);
        verify(playerRepository).saveAll(captor.capture());

        List<Player> savedPlayers = captor.getValue();
        
        assertEquals(1, savedPlayers.size(), "Dovrebbe salvare solo la riga valida");
        assertEquals("Roberto Baggio", savedPlayers.get(0).getName());
    }
}